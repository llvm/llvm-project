#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script for building the LLVM installer on Windows,
    used for the releases at https://github.com/llvm/llvm-project/releases

.DESCRIPTION
    Builds LLVM release packages for Windows (x64, arm64).
    Performs a 2-stage build with a ThinLTO final stage and optional PGO.
    Can bootstrap a fresh VM by installing all prerequisites.

    By default, builds x64 using the local source tree and auto-detected
    Python. Use -DownloadSource to download a tagged release tarball
    instead.

    Build steps (in order):
      1. libxml2   - Build libxml2, zlib, and zstd (inside stage 1 directory)
      2. stage1    - Build the stage 1 bootstrap compiler with the host compiler
      3. pgo       - PGO instrumented build + training + profile merge
      4. stage2    - Build the stage 2 self-hosted compiler with stage 1 clang
      5. package   - Create WiX MSI installer
      6. tarball   - Generate full install tarball

.PARAMETER Version
    LLVM version string (e.g. "19.1.0"). If omitted, auto-detected from
    the source tree.

.PARAMETER x64
    Build for x64 (64-bit). This is the default if no architecture is specified.

.PARAMETER arm64
    Build for ARM64 (AArch64).

.PARAMETER DownloadSource
    Download and extract the tagged release source tarball from GitHub
    instead of using the local source tree. Requires -Version.

.PARAMETER ForceMSVC
    Force using MSVC (cl.exe) as the stage 0 host compiler instead of the
    official LLVM release. Skips the LLVM release prerequisite.

.PARAMETER FastBuild
    Build only the stage 1 tools needed by stage 2 and skip the remaining
    stage 1 targets, stage 1 tests, and PGO. Stage 2 tests still run.

.PARAMETER InstallPrerequisites
    Install build prerequisites (NetFx3, Visual Studio, official LLVM release,
    CMake, Python and its psutil module, etc.) before building.
    When used with -arm64, also installs ARM64 Python for LLDB.
    Requests administrator elevation once for the entire prerequisite
    installation when the current shell is not elevated.

.PARAMETER Unattended
    Never prompt for input. Missing prerequisites cause the build to fail
    unless -InstallPrerequisites is also specified. An existing build
    directory must be resumed with -StartAt or removed beforehand. Run
    from an elevated shell when using -InstallPrerequisites.

.PARAMETER StartAt
    Resume a previous build from a specific step. Artifacts from earlier
    steps must already exist on disk. The step being restarted gets a
    clean directory; earlier steps are skipped entirely.

    Valid steps: libxml2, stage1, pgo, stage2, package, tarball

    Pass an empty string (-StartAt "") or "?" (-StartAt "?") to display
    the list of available steps and exit.

.PARAMETER Help
    Display this help message and exit.

.EXAMPLE
    .\build_llvm_release.ps1
    Full x64 build using the local source tree.

.EXAMPLE
    .\build_llvm_release.ps1 -arm64
    Build for ARM64.

.EXAMPLE
    .\build_llvm_release.ps1 -Version 19.1.0 -DownloadSource
    Download version 19.1.0 sources and build x64.

.EXAMPLE
    .\build_llvm_release.ps1 -InstallPrerequisites -x64
    Install prerequisites, then do a full x64 build.

.EXAMPLE
    .\build_llvm_release.ps1 -Unattended -InstallPrerequisites
    From an elevated shell, install prerequisites and build without prompts.

.EXAMPLE
    .\build_llvm_release.ps1 -x64 -StartAt stage2
    Resume from stage 2 (reuses the stage 1 bootstrap compiler and PGO profile).

.EXAMPLE
    .\build_llvm_release.ps1 -x64 -StartAt package
    Resume at WiX MSI packaging, then regenerate the portable archive.

.EXAMPLE
    .\build_llvm_release.ps1 -StartAt ""
    Display the list of available build steps and exit.

.NOTES
    Python is auto-detected from PATH for x64 builds. For ARM64 builds,
    the script probes standard install locations for ARM64 Python. Use
    -InstallPrerequisites to install it automatically.

    Environment variables:
      LLVM_NINJA_OVERRIDE  - Override the ninja binary and optionally provide
                             extra flags. The first token is the executable,
                             remaining tokens are prepended to every ninja
                             invocation.
                             Example: LLVM_NINJA_OVERRIDE="myninja.exe --flag1 --flag2"
#>

param(
    [string]$Version,
    [switch]$x64,
    [switch]$arm64,
    [switch]$DownloadSource,
    [switch]$ForceMSVC,
    [switch]$FastBuild,
    [switch]$InstallPrerequisites,
    [switch]$Unattended,
    [switch]$PrerequisitesOnly,  # Internal mode for the elevated installer.
    [string]$StartAt,
    [switch]$Help
)

#===============================================================================
# PowerShell 7 self-relaunch
#
# This script requires PowerShell 7+ features (ternary operator, null-
# coalescing, improved error handling, etc.). If we detect we are running
# under Windows PowerShell 5.x, we attempt to find or install PowerShell 7
# and re-launch ourselves under it, forwarding all original arguments.
#===============================================================================
if ($PSVersionTable.PSVersion.Major -lt 7) {
    # Try to find pwsh in PATH first.
    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $pwsh) {
        # Check the default install location.
        $defaultPath = "$env:ProgramFiles\PowerShell\7\pwsh.exe"
        if (Test-Path $defaultPath) {
            $pwsh = Get-Item $defaultPath
        }
    }

    if (-not $pwsh) {
        if ($Unattended) {
            Write-Error "PowerShell 7 is required. Install it before running in unattended mode."
            exit 1
        }
        Write-Host "PowerShell 7 is required but not found." -ForegroundColor Yellow
        Write-Host "Attempting to install via winget..." -ForegroundColor Yellow
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) {
            Write-Error "Cannot install PowerShell 7: winget is not available.`nPlease install PowerShell 7 manually from https://aka.ms/powershell-release?tag=stable"
            exit 1
        }
        winget install --id Microsoft.PowerShell --source winget --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install PowerShell 7 via winget (exit code $LASTEXITCODE)."
            exit 1
        }
        # Refresh PATH so we can find the newly installed pwsh.
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                     [System.Environment]::GetEnvironmentVariable("Path", "User")
        $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
        if (-not $pwsh) {
            $defaultPath = "$env:ProgramFiles\PowerShell\7\pwsh.exe"
            if (Test-Path $defaultPath) {
                $pwsh = Get-Item $defaultPath
            }
        }
        if (-not $pwsh) {
            Write-Error "PowerShell 7 was installed but pwsh.exe could not be found in PATH."
            exit 1
        }
        Write-Host "PowerShell 7 installed successfully." -ForegroundColor Green
    }

    # Re-launch this script under PowerShell 7, forwarding all arguments.
    # Note: avoid PS7-only syntax (??  ternary, etc.) in this block since
    # it executes under PS5 (PowerShell 5).
    $pwshPath = if ($pwsh.Source) { $pwsh.Source } else { $pwsh.Path }
    Write-Host "Re-launching under PowerShell 7 ($pwshPath)..." -ForegroundColor Cyan
    $forwardArgs = @()
    foreach ($entry in $PSBoundParameters.GetEnumerator()) {
        if ($entry.Value -is [System.Management.Automation.SwitchParameter]) {
            if ($entry.Value.IsPresent) { $forwardArgs += "-$($entry.Key)" }
        } else {
            $forwardArgs += "-$($entry.Key)"
            # Windows PowerShell 5.1 drops empty-string arguments when calling
            # native programs; a literal pair of quotes survives as "".
            $stringValue = [string]$entry.Value
            $forwardArgs += if ($stringValue -eq '') { '""' } else { $stringValue }
        }
    }
    & $pwshPath -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath @forwardArgs
    exit $LASTEXITCODE
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:ReleaseScriptPath = $PSCommandPath
$script:MinimumCMakeVersion = [version]'3.31.0'

# Save the console mode so we can restore it at exit. Child processes
# (cmake, ninja, link.exe, etc.) sometimes disable virtual terminal
# processing and don't restore it, which breaks arrow keys and ESC in
# the parent shell after the script finishes.
$script:SavedConsoleMode = $null
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ConsoleMode {
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr GetStdHandle(int nStdHandle);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
    const int STD_INPUT_HANDLE = -10;
    public static uint Get() {
        uint mode;
        GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), out mode);
        return mode;
    }
    public static void Set(uint mode) {
        SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), mode);
    }
}
'@ -ErrorAction SilentlyContinue
    $script:SavedConsoleMode = [ConsoleMode]::Get()
} catch {
    # Non-fatal; we just won't be able to restore the console mode.
}

# Default to x64 if no architecture specified
if (-not $x64 -and -not $arm64) {
    $x64 = $true
}

#===============================================================================
# Step-resume support
#===============================================================================

$script:StepOrder   = @('libxml2','stage1','pgo','stage2','package','tarball')

if ($PSBoundParameters.ContainsKey('StartAt')) {
    if (-not $StartAt -or $StartAt -eq '?') {
        Write-Host "Available -StartAt steps (in order):" -ForegroundColor Cyan
        for ($i = 0; $i -lt $script:StepOrder.Count; $i++) {
            Write-Host "  $($i + 1). $($script:StepOrder[$i])"
        }
        exit 0
    }
    if ($StartAt -notin $script:StepOrder) {
        Write-Error "Invalid -StartAt value: '$StartAt'. Valid steps: $($script:StepOrder -join ', ')"
        exit 1
    }
}

$script:StartAtStep = if ($StartAt) { $StartAt } else { $null }

function Test-ShouldRun {
    <#
    .SYNOPSIS
        Returns $true if the given step should execute (i.e. it is at or after
        the -StartAt step). When -StartAt is not set, always returns $true.
    #>
    param([Parameter(Mandatory)][string]$Step)
    if (-not $script:StartAtStep) { return $true }
    return $script:StepOrder.IndexOf($Step) -ge $script:StepOrder.IndexOf($script:StartAtStep)
}

function Test-IsStartStep {
    <#
    .SYNOPSIS
        Returns $true if the given step is exactly the -StartAt step (the step
        whose directory should be cleaned before a fresh rebuild).
    #>
    param([Parameter(Mandatory)][string]$Step)
    return ($script:StartAtStep -and $script:StartAtStep -eq $Step)
}

function Assert-PathExists {
    <#
    .SYNOPSIS
        Validates that a path exists on disk. Used when skipping steps to
        ensure the artifacts from a prior run are still present.
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [string]$Description = $Path
    )
    if (-not (Test-Path $Path)) {
        Write-Error "Required artifact missing: $Description`n  Path: $Path`n  Hint: run the earlier build steps first, or use a different -StartAt value."
        exit 1
    }
}

function Remove-StepDirectory {
    <#
    .SYNOPSIS
        Removes a directory (or file) if it exists, printing a message.
        Used to clean a step's build directory before restarting it.
    #>
    param([Parameter(Mandatory)][string]$Path)
    if (Test-Path $Path) {
        Write-SubStep "Cleaning: $Path"
        Remove-Item -Recurse -Force $Path
    }
}

# Ninja override: allow using a custom ninja binary with extra flags.
# Usage: $env:LLVM_NINJA_OVERRIDE = "myninja.exe --flag1 --flag2"
if ($env:LLVM_NINJA_OVERRIDE) {
    $tokens = @($env:LLVM_NINJA_OVERRIDE.Trim() -split '\s+')
    $script:NinjaCommand  = $tokens[0]
    [string[]]$script:NinjaExtraArgs = if ($tokens.Count -gt 1) { $tokens[1..($tokens.Count - 1)] } else { @() }
    Write-Host "Using custom ninja: $($script:NinjaCommand)" -ForegroundColor Cyan
    if ($script:NinjaExtraArgs.Count -gt 0) {
        Write-Host "  Extra ninja args: $($script:NinjaExtraArgs -join ' ')" -ForegroundColor Cyan
    }
} else {
    $script:NinjaCommand  = 'ninja'
    $script:NinjaExtraArgs = @()
}

# Filter out tests that are known to fail.
$env:LIT_FILTER_OUT = "gh110231.cpp|crt_initializers.cpp|init-order-atexit.cpp|use_after_return_linkage.cpp|initialization-bug.cpp|initialization-bug-no-global.cpp|trace-malloc-unbalanced.test|trace-malloc-2.test|TraceMallocTest|TestLockFileExclusive"

#===============================================================================
# Utility functions
#===============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host "`n==== $Message ====`n" -ForegroundColor Cyan
}

function Write-SubStep {
    param([string]$Message)
    Write-Host "  -- $Message" -ForegroundColor DarkCyan
}

function Write-CMakeCacheFile {
    <#
    .SYNOPSIS
        Writes CMake cache assignments to an initial cache script file to avoid
        exceeding the Windows command-line length limit.
    .DESCRIPTION
        Accepts VAR[:TYPE]=VALUE entries and writes each as a
        set(VAR VALUE CACHE <type> "") line in a temporary .cmake file.
        Returns a hashtable with:
          CacheFile  - the path to the generated file
          OtherFlags - any non-assignment arguments in the input array
    #>
    param(
        [Parameter(Mandatory)]
        [string[]]$Flags,
        [string]$FileName = 'cache_flags.cmake'
    )
    $cacheLines = @()
    $otherFlags = @()
    foreach ($f in $Flags) {
        if ($f -match '^([A-Za-z_][A-Za-z0-9_]*)(?::([^=]*))?=(.*)$') {
            $varName  = $Matches[1]
            $varType  = if ($Matches[2]) { $Matches[2] } else { 'STRING' }
            $varValue = ($Matches[3] -replace '"', '').Replace('\', '/')   # strip quotes, use forward slashes
            $cacheLines += "set($varName `"$varValue`" CACHE $varType `"`" FORCE)"
        } else {
            $otherFlags += $f
        }
    }
    $cachePath = (Join-Path $PWD $FileName).Replace('\', '/')
    $cacheLines -join "`n" | Set-Content -Path $cachePath -Encoding UTF8
    return @{
        CacheFile  = $cachePath
        OtherFlags = $otherFlags
    }
}

function Invoke-NativeCommand {
    <#
    .SYNOPSIS
        Runs a native command and throws on non-zero exit code.
    .NOTES
        This is intentionally a simple function (no [Parameter()] attributes
        and no [CmdletBinding()]) so that PowerShell does NOT inject common
        parameters (-Confirm, -OutVariable, -ErrorAction, etc.). Those
        common parameters collide with native flags like -C, -O, -E, etc.
    #>
    $Command = $args[0]
    $Arguments = @()
    if ($args.Count -gt 1) {
        $Arguments = @($args[1..($args.Count - 1)])
    }
    Write-Host "+ $Command $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

function Test-ExecutableVersion {
    param([string]$Path)

    try {
        & $Path --version *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-ForwardSlashPath {
    <#
    .SYNOPSIS
        Convert backslashes to forward slashes (CMake expects this for compiler paths).
    #>
    param([string]$Path)
    return $Path.Replace('\', '/')
}

function Test-FileChecksum {
    <#
    .SYNOPSIS
        Verifies a downloaded file's hash and throws if it doesn't match.
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Algorithm,
        [Parameter(Mandatory)][string]$ExpectedHash
    )
    $actual = (Get-FileHash -Path $Path -Algorithm $Algorithm).Hash
    if ($actual -ne $ExpectedHash.ToUpperInvariant()) {
        throw "Checksum mismatch for ${Path}:`n  expected $ExpectedHash`n  actual   $actual"
    }
    Write-SubStep "Checksum OK ($Algorithm): $Path"
}

#===============================================================================
# Version detection
#===============================================================================

function Get-LLVMVersionFromSource {
    <#
    .SYNOPSIS
        Extracts LLVM version from cmake/Modules/LLVMVersion.cmake.
    #>
    param([string]$SourceDir)
    $versionFile = Join-Path $SourceDir 'cmake' 'Modules' 'LLVMVersion.cmake'
    if (-not (Test-Path $versionFile)) {
        throw "Cannot find LLVMVersion.cmake at: $versionFile"
    }
    $content = Get-Content $versionFile -Raw
    $major = if ($content -match 'LLVM_VERSION_MAJOR\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_MAJOR" }
    $minor = if ($content -match 'LLVM_VERSION_MINOR\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_MINOR" }
    $patch = if ($content -match 'LLVM_VERSION_PATCH\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_PATCH" }
    $suffix = if ($content -match 'LLVM_VERSION_SUFFIX\s+(\S+)\)') { $Matches[1] } else { '' }
    # Don't include the "git" development suffix in the release version
    if ($suffix -eq 'git') { $suffix = '' }
    return @{
        Major = $major
        Minor = $minor
        Patch = $patch
        Suffix = $suffix
        Full = "${major}.${minor}.${patch}${suffix}"
    }
}

#===============================================================================
# Prerequisite validation
#===============================================================================

function Find-Wix314Bin {
    $binDirs = @(
        "${env:ProgramFiles(x86)}\WiX Toolset v3.14\bin"
        "$env:ProgramFiles\WiX Toolset v3.14\bin"
    )
    $binDirs += @(Get-Command candle -All -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { Split-Path -Parent $_.Source })

    foreach ($binDir in ($binDirs | Select-Object -Unique)) {
        $candlePath = Join-Path $binDir 'candle.exe'
        $lightPath = Join-Path $binDir 'light.exe'
        if (-not (Test-Path -LiteralPath $candlePath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $lightPath -PathType Leaf)) { continue }
        if ((Get-Item -LiteralPath $candlePath).VersionInfo.FileVersion -notlike '3.14.*' -or
            (Get-Item -LiteralPath $lightPath).VersionInfo.FileVersion -notlike '3.14.*') {
            continue
        }
        try {
            & $candlePath -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
            & $lightPath -? *> $null
            if ($LASTEXITCODE -eq 0) { return $binDir }
        } catch {
            # Try the next WiX installation location.
        }
    }
    return $null
}

function Get-CMakeVersion {
    param([Parameter(Mandatory)][string]$Path)

    try {
        $versionOutput = (& $Path --version 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0 -or
            $versionOutput -notmatch '(?m)^cmake version (\d+(?:\.\d+){1,3})\s*$') {
            return $null
        }
        return [version]$Matches[1]
    } catch {
        return $null
    }
}

function Find-CMakeExecutable {
    # Prefer the first usable version on PATH, then the standard CMake
    # installation, and finally CMake bundled with any Visual Studio instance.
    $candidates = @()
    foreach ($pathDir in ($env:PATH -split ';')) {
        if ($pathDir) {
            $candidates += Join-Path $pathDir 'cmake.exe'
        }
    }

    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'
    }

    $vsInstallPaths = @()
    if ($env:VSINSTALLDIR) {
        $vsInstallPaths += $env:VSINSTALLDIR
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        try {
            $vsInstallPaths += @(& $vswhere -all -products '*' -property installationPath 2>$null)
        } catch {
            # CMake may still be available through PATH or the standard install.
        }
    }
    foreach ($vsInstallPath in ($vsInstallPaths | Where-Object { $_ } | Select-Object -Unique)) {
        $candidates += Join-Path $vsInstallPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }

        $version = Get-CMakeVersion -Path $candidate
        if ($null -eq $version -or $version -lt $script:MinimumCMakeVersion) {
            if ($null -ne $version) {
                Write-SubStep "Ignoring CMake $version at $candidate; LLVM requires $($script:MinimumCMakeVersion) or newer."
            }
            continue
        }

        # Put the selected directory first so every later `cmake` invocation
        # uses the validated executable, even if an older one appeared earlier.
        $binDir = Split-Path -Parent $candidate
        $otherPaths = @($env:PATH -split ';' | Where-Object {
            $_ -and $_.TrimEnd('\') -ine $binDir.TrimEnd('\')
        })
        $env:PATH = (@($binDir) + $otherPaths) -join ';'
        Write-SubStep "CMake ${version}: $candidate"
        return $candidate
    }

    return $null
}

function Find-SevenZipExecutable {
    $candidates = @()
    $command = Get-Command 7z -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($programFiles in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($programFiles) {
            $candidates += Join-Path $programFiles '7-Zip\7z.exe'
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
        } catch {
            continue
        }

        # An existing installation may not have added 7z.exe to PATH.
        if (-not $command -or $command.Source -ne $candidate) {
            $binDir = Split-Path -Parent $candidate
            $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $binDir })
            $env:PATH = (@($binDir) + $otherPaths) -join ';'
            Write-SubStep "Using 7-Zip from: $binDir"
        }
        return $candidate
    }
    return $null
}

function Find-MakeExecutable {
    $candidates = @()
    $command = Get-Command make -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($programFiles in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($programFiles) {
            $candidates += Join-Path $programFiles 'GnuWin32\bin\make.exe'
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
            -not (Test-ExecutableVersion -Path $candidate)) { continue }

        if (-not $command -or $command.Source -ne $candidate) {
            $binDir = Split-Path -Parent $candidate
            $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $binDir })
            $env:PATH = (@($binDir) + $otherPaths) -join ';'
            Write-SubStep "Using GNU Make from: $binDir"
        }
        return $candidate
    }
    return $null
}

function Find-OfficialLlvmBin {
    $binDir = Join-Path $env:ProgramFiles 'LLVM\bin'
    $clangCl = Join-Path $binDir 'clang-cl.exe'
    $lldLink = Join-Path $binDir 'lld-link.exe'
    if ((Test-Path -LiteralPath $clangCl -PathType Leaf) -and
        (Test-Path -LiteralPath $lldLink -PathType Leaf) -and
        (Test-ExecutableVersion -Path $clangCl) -and
        (Test-ExecutableVersion -Path $lldLink)) {
        return $binDir
    }
    return $null
}

function Get-MissingPrerequisites {
    <#
    .SYNOPSIS
        Applies PATH fixups, checks required build tools, and returns the
        names of any that are missing.
    #>
    Write-Step "Validating prerequisites"

    # Fix up PATH for tools that install to well-known non-PATH locations.
    $pathFixups = @(
        "$env:ProgramFiles\Git\usr\bin"                     # sh, bash, sed, grep (needed by tests)
        "${env:ProgramFiles(x86)}\GnuWin32\bin"             # make (GnuWin32 default)
        "$env:ProgramFiles\GnuWin32\bin"                    # make (alt location)
    )
    foreach ($dir in $pathFixups) {
        if ((Test-Path $dir) -and ($env:PATH -notlike "*$dir*")) {
            $env:PATH = "$env:PATH;$dir"
            Write-SubStep "Added to PATH: $dir"
        }
    }

    $cmakePath = Find-CMakeExecutable

    # CPack finds WiX through PATH. An older candle/light pair can appear
    # before the installed 3.14 tools (for example, in an Unreal SDK).
    $wixBin = Find-Wix314Bin
    if ($wixBin) {
        $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $wixBin })
        $env:PATH = (@($wixBin) + $otherPaths) -join ';'
        Write-SubStep "Using WiX 3.14 from: $wixBin"
    }

    # Check all required tools.
    $required = @(
        @{ Name = "Ninja";      Cmd = "ninja" }
        @{ Name = "Python";     Cmd = "python" }
        @{ Name = "7-Zip";      Cmd = "7z" }
        @{ Name = "WiX candle"; Cmd = "candle" }
        @{ Name = "WiX light";  Cmd = "light" }
        @{ Name = "Git";        Cmd = "git" }
        @{ Name = "SWIG";       Cmd = "swig" }
        @{ Name = "Perl";       Cmd = "perl" }
        @{ Name = "Make";       Cmd = "make" }
    )

    $missing = @()
    foreach ($tool in $required) {
        $cmd = Get-Command $tool.Cmd -ErrorAction SilentlyContinue
        $toolPath = if ($cmd) { $cmd.Source } else { $null }
        # Windows may expose a python.exe App Execution Alias that is not an
        # installed interpreter.
        if ($cmd -and $tool.Cmd -eq 'python' -and
            -not (Test-ExecutableVersion -Path $cmd.Source)) {
            $toolPath = $null
        }
        if ($tool.Cmd -eq '7z') {
            $toolPath = Find-SevenZipExecutable
        }
        if ($tool.Cmd -eq 'make') {
            $toolPath = Find-MakeExecutable
        }
        # These tools are deprecated and removed in newer version of WiX.
        if ($tool.Cmd -in @('candle', 'light') -and -not $wixBin) {
            $toolPath = $null
        }
        if ($toolPath) {
            Write-SubStep "$($tool.Name): $toolPath"
        } else {
            $missing += $tool.Name
        }
    }

    if (-not $cmakePath) {
        $missing += "CMake $($script:MinimumCMakeVersion) or newer"
    }

    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if ($llvmBin) {
            Write-SubStep "LLVM release (clang-cl and lld-link): $llvmBin"
        } else {
            $missing += 'LLVM release (clang-cl and lld-link)'
        }
    }

    # Lit disables the Python user site when running tests. Check the Python
    # installation itself so timeouts and tests requiring them stay available.
    $hostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if ($hostPython) {
        if (Test-PythonPsutil -PythonExecutable $hostPython) {
            Write-SubStep "Python psutil: $hostPython"
        } else {
            $missing += 'Python psutil'
        }
    }
    if ($arm64) {
        $arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($arm64Python -and $arm64Python -ne $hostPython) {
            if (Test-PythonPsutil -PythonExecutable $arm64Python) {
                Write-SubStep "Python psutil (ARM64): $arm64Python"
            } else {
                $missing += 'Python psutil (ARM64)'
            }
        }
    }

    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5): installed'
    } else {
        $missing += 'NetFx3 (.NET Framework 3.5)'
    }

    return $missing
}

function Assert-SevenZipSymlinkSupport {
    # 7-Zip 21+ extracts symlinks from LLVM's source archive. Windows must
    # allow symlink creation when -DownloadSource is selected.
    $banner = (& 7z 2>&1 | Select-Object -First 2) -join ' '
    if ($banner -notmatch '7-Zip\s+(\d+)\.' -or [int]$Matches[1] -lt 21) {
        return
    }

    $linkPath = Join-Path ([IO.Path]::GetTempPath()) ("llvm-release-" + [guid]::NewGuid())
    try {
        New-Item -ItemType SymbolicLink -Path $linkPath -Target $PWD.Path -ErrorAction Stop | Out-Null
    } catch {
        throw "7-Zip 21+ needs permission to create symlinks when extracting the source archive. Run elevated, enable Developer Mode, or use 7-Zip 20.x."
    } finally {
        Remove-Item -LiteralPath $linkPath -Force -ErrorAction SilentlyContinue
    }
}

#===============================================================================
# Prerequisite installation (for fresh VMs)
#===============================================================================

function Test-IsElevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-NetFx3Installed {
    # The .NET Framework 3.5 installer records this value on both Windows
    # Server and desktop Windows. WiX 3.14 needs the feature to run.
    $key = 'HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v3.5'
    $framework = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
    return $null -ne $framework -and $framework.Install -eq 1
}

function Install-NetFx3 {
    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5) is already installed.'
        return
    }

    Write-SubStep 'Installing NetFx3 (.NET Framework 3.5)...'
    $installationType = (Get-ItemProperty `
        -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').InstallationType
    if ($installationType -like 'Server*') {
        # Windows Server exposes NetFx3 as the NET-Framework-Core server role.
        # Use the built-in Windows PowerShell to load ServerManager reliably,
        # even when this release script is running under PowerShell 7.
        $windowsPowerShell = Join-Path $env:SystemRoot `
            'System32\WindowsPowerShell\v1.0\powershell.exe'
        $serverCommand = @'
$ErrorActionPreference = 'Stop'
Import-Module ServerManager
$feature = Get-WindowsFeature -Name NET-Framework-Core
if (-not $feature) { throw 'NET-Framework-Core is not available on this server.' }
if (-not $feature.Installed) {
    $result = Install-WindowsFeature -Name NET-Framework-Core
    if (-not $result.Success) {
        throw "Install-WindowsFeature failed: $($result.ExitCode)"
    }
    if ($result.RestartNeeded -eq 'Yes') { exit 3010 }
}
'@
        $encodedCommand = [Convert]::ToBase64String(
            [Text.Encoding]::Unicode.GetBytes($serverCommand))
        & $windowsPowerShell -NoProfile -NonInteractive -EncodedCommand $encodedCommand
        if ($LASTEXITCODE -eq 3010) {
            throw 'NetFx3 installation requires a restart. Restart Windows and rerun the release script.'
        }
        if ($LASTEXITCODE -ne 0) {
            throw "NetFx3 installation failed (exit code $LASTEXITCODE). If Windows cannot find the feature files, install NET-Framework-Core from matching Windows Server media and rerun the release script."
        }
    } else {
        $result = Enable-WindowsOptionalFeature -Online -FeatureName NetFx3 `
            -All -NoRestart -ErrorAction Stop
        if ($result.RestartNeeded) {
            throw 'NetFx3 installation requires a restart. Restart Windows and rerun the release script.'
        }
    }

    if (-not (Test-NetFx3Installed)) {
        throw 'NetFx3 installation completed, but .NET Framework 3.5 is not registered as installed.'
    }
    Write-SubStep 'NetFx3 installed.'
}

function Update-PathAfterPrerequisites {
    # The elevated installer runs in a separate process, so refresh the
    # caller's PATH from the machine and user environment after it exits.
    $machinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $env:PATH = "$machinePath;$userPath"
}

function Invoke-ElevatedPrerequisiteInstallation {
    $scriptPath = $script:ReleaseScriptPath.Replace("'", "''")
    $invocation = "& '$scriptPath' -InstallPrerequisites -PrerequisitesOnly"
    if ($arm64) { $invocation += ' -arm64' }
    if ($ForceMSVC) { $invocation += ' -ForceMSVC' }
    # Keep the elevated window open so the user can review the result.
    $pauseCommand = if (-not $Unattended) {
        "Read-Host 'Press Enter to close this window' | Out-Null"
    } else {
        ''
    }
    $workerCommand = @"
`$exitCode = 0
try {
    $invocation
} catch {
    Write-Host "Prerequisite installation failed:" -ForegroundColor Red
    Write-Host (`$_ | Out-String) -ForegroundColor Red
    `$exitCode = 1
}
$pauseCommand
exit `$exitCode
"@
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($workerCommand))

    Write-SubStep 'Requesting administrator access to install prerequisites...'
    try {
        $process = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encodedCommand) `
            -Verb RunAs -Wait -PassThru
    } catch {
        throw "Could not start elevated prerequisite installation: $_"
    }
    if ($process.ExitCode -ne 0) {
        throw "Elevated prerequisite installation failed (exit code $($process.ExitCode))."
    }
}

function Find-WingetExecutable {
    $candidates = @()
    $winget = Get-Command winget -CommandType Application -ErrorAction SilentlyContinue
    if ($winget) { $candidates += $winget.Source }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
    }

    # On fresh Windows Server images the App Installer alias may be absent
    # from the elevated PATH. Discover the installed package directly too.
    try {
        $packages = @(Get-AppxPackage -AllUsers -Name Microsoft.DesktopAppInstaller -ErrorAction Stop)
        foreach ($package in ($packages | Sort-Object Version -Descending)) {
            if ($package.InstallLocation) {
                $candidates += Join-Path $package.InstallLocation 'winget.exe'
            }
        }
    } catch {
        # Appx discovery is not available on every Windows image.
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {
            # Try the next location.
        }
    }
    throw "winget cannot run in the elevated shell. Checked: $($candidates -join ', '). Ensure App Installer is available to the account used by UAC."
}

function Assert-WixToolsUsable {
    $binDir = Find-Wix314Bin
    if (-not $binDir) {
        throw 'WiX 3.14 was installed, but candle.exe and light.exe could not run.'
    }
    Write-SubStep "WiX tools verified: $binDir"
}

function Install-Prerequisites {
    if (-not (Test-IsElevated)) {
        if ($Unattended) {
            throw "Unattended prerequisite installation requires administrator privileges. Start an elevated shell before using -Unattended -InstallPrerequisites."
        }
        Invoke-ElevatedPrerequisiteInstallation
        Update-PathAfterPrerequisites
        if (-not (Find-CMakeExecutable)) {
            throw "CMake $($script:MinimumCMakeVersion) or newer could not be found after prerequisite installation."
        }
        Install-PythonPsutilForRequestedArchitectures
        return
    }

    Write-Step "Installing prerequisites"

    Install-NetFx3
    $wingetPath = Find-WingetExecutable

    # Version requirements for the release build:
    #   - CMake 3.31 or newer
    #   - NetFx3 (.NET Framework 3.5) for WiX Toolset 3.14
    #   - Python 3.13 for all architectures
    #   - SWIG 4 or newer for LLDB (install the latest available WinGet version)
    #   - WiX Toolset 3.14 for MSI packaging
    #   - Perl is needed for the OpenMP run-time
    #   - GNU Make is needed by LLDB API tests
    #   - 7-Zip 21.x+ needs permission to create source archive symlinks
    #   - LLVM.LLVM supplies the stage 0 clang-cl and lld-link tools
    $tools = @(
        @{ Name = "cmake";      WingetId = "Kitware.CMake";                 Version = "";            Verify = "cmake" }
        @{ Name = "ninja";      WingetId = "Ninja-build.Ninja";             Version = "";            Verify = "ninja" }
        @{ Name = "python3.13"; WingetId = "Python.Python.3.13";            Version = "";            Verify = "python" }
        @{ Name = "7z";         WingetId = "7zip.7zip";                     Version = "";            Verify = "7z" }
        @{ Name = "wix";        WingetId = "WiXToolset.WiXToolset";         Version = "3.14.1.8722"; Verify = "candle" }
        @{ Name = "git";        WingetId = "Git.Git";                       Version = "";            Verify = "git" }
        @{ Name = "swig";       WingetId = "SWIG.SWIG";                     Version = "";            Verify = "swig" }
        @{ Name = "perl";       WingetId = "StrawberryPerl.StrawberryPerl"; Version = "";            Verify = "perl" }
        @{ Name = "make";       WingetId = "GnuWin32.Make";                 Version = "";            Verify = "make" }
    )

    foreach ($tool in $tools) {
        $cmd = Get-Command $tool.Verify -ErrorAction SilentlyContinue
        if ($cmd -and $tool.Verify -eq 'python' -and
            -not (Test-ExecutableVersion -Path $cmd.Source)) {
            $cmd = $null
        }
        $existingPath = if ($tool.Name -eq 'cmake') { Find-CMakeExecutable }
            elseif ($tool.Name -eq 'wix') { Find-Wix314Bin }
            elseif ($tool.Name -eq '7z') { Find-SevenZipExecutable }
            elseif ($tool.Name -eq 'make') { Find-MakeExecutable }
            elseif ($cmd) { $cmd.Source }
        if ($existingPath) {
            Write-SubStep "$($tool.Name) is already installed: $existingPath"
        } else {
            $versionArg = if ($tool.Version) { @("--version", $tool.Version) } else { @() }
            $versionLabel = if ($tool.Version) { " $($tool.Version)" } else { '' }
            Write-SubStep "Installing $($tool.Name) ($($tool.WingetId)$versionLabel)..."
            $installArgs = @('install', '--id', $tool.WingetId) + $versionArg +
                @('--accept-package-agreements', '--accept-source-agreements', '--silent')
            & $wingetPath @installArgs
            $installExitCode = $LASTEXITCODE
            if ($tool.Name -eq 'wix' -and $installExitCode -eq -1978334960) {
                # winget can fail while checking its NetFx3 dependency on
                # Server 2025 even after the feature is installed. The feature
                # was installed and verified above; bypass only this check.
                Write-Warning 'winget could not verify the installed NetFx3 feature. Retrying WiX with --force.'
                & $wingetPath @installArgs --force
                $installExitCode = $LASTEXITCODE
            }
            if ($tool.Name -eq '7z') {
                $sevenZipPath = Find-SevenZipExecutable
                if ($sevenZipPath) {
                    if ($installExitCode -ne 0) {
                        Write-Warning "winget exited with code $installExitCode, but 7-Zip is usable at $sevenZipPath. Continuing."
                    }
                    Write-SubStep "7-Zip verified: $sevenZipPath"
                    continue
                }
            }
            if ($tool.Name -eq 'make') {
                $makePath = Find-MakeExecutable
                if ($makePath) {
                    if ($installExitCode -ne 0) {
                        Write-Warning "winget exited with code $installExitCode, but GNU Make is usable at $makePath. Continuing."
                    }
                    Write-SubStep "GNU Make verified: $makePath"
                    continue
                }
            }
            if ($tool.Name -eq 'cmake') {
                if (-not (Find-CMakeExecutable)) {
                    throw "CMake installation finished, but version $($script:MinimumCMakeVersion) or newer could not be found."
                }
                continue
            }
            if ($installExitCode -ne 0) {
                if ($tool.Name -eq 'wix') {
                    throw "Failed to install WiX via winget (exit code $installExitCode). NetFx3 is installed; inspect winget's error above."
                }
                throw "Failed to install $($tool.Name) via winget (exit code $installExitCode)."
            }
            if ($tool.Name -eq '7z') {
                throw '7-Zip installation finished, but 7z.exe could not be found or run.'
            }
            if ($tool.Name -eq 'make') {
                throw 'GNU Make installation finished, but make.exe could not be found or run.'
            }
            if ($tool.Name -eq 'wix') { Assert-WixToolsUsable }
        }
    }

    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if ($llvmBin) {
            Write-SubStep "LLVM release is already installed: $llvmBin"
        } else {
            Write-SubStep 'Installing LLVM release (LLVM.LLVM)...'
            & $wingetPath install --id LLVM.LLVM --exact --source winget `
                --accept-package-agreements --accept-source-agreements --silent
            $installExitCode = $LASTEXITCODE
            $llvmBin = Find-OfficialLlvmBin
            if (-not $llvmBin) {
                throw "LLVM installation did not provide usable clang-cl.exe and lld-link.exe in $env:ProgramFiles\LLVM\bin (winget exit code $installExitCode)."
            }
            if ($installExitCode -ne 0) {
                Write-Warning "winget exited with code $installExitCode, but LLVM tools are usable at $llvmBin. Continuing."
            }
            Write-SubStep "LLVM release verified: $llvmBin"
        }
    }

    # Install per-architecture Python for cross-arch builds (LLDB needs matching Python).
    # The main Python install above covers the host architecture (typically x64).
    $archPython = @()
    if ($arm64) { $archPython += @{ Arch = 'arm64'; Suffix = '-arm64' } }
    foreach ($ap in $archPython) {
        $basePath = "$env:LOCALAPPDATA\Programs\Python"
        $existing = if (Test-Path $basePath) {
            Get-ChildItem -Path $basePath -Directory -Filter "Python3*$($ap.Suffix)" |
                Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') }
        } else { $null }
        if ($existing) {
            Write-SubStep "Python for $($ap.Arch) already installed: $($existing[0].FullName)"
        } else {
            Write-SubStep "Installing Python for $($ap.Arch) (Python.Python.3.13 --architecture $($ap.Arch))..."
            & $wingetPath install --id Python.Python.3.13 --architecture $($ap.Arch) --accept-package-agreements --accept-source-agreements --silent
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install Python for $($ap.Arch). LLDB may not work for this architecture."
            }
        }
    }

    # Visual Studio Build Tools - special handling
    Install-VisualStudio -WingetPath $wingetPath

    # Get-MissingPrerequisites will add non-standard paths such as GnuWin32.
    Update-PathAfterPrerequisites
    if (-not (Find-CMakeExecutable)) {
        throw "CMake $($script:MinimumCMakeVersion) or newer could not be found after prerequisite installation."
    }
    Install-PythonPsutilForRequestedArchitectures
}

function Install-VisualStudio {
    param([Parameter(Mandatory)][string]$WingetPath)

    # Components required by LLVM:
    #   - VCTools workload: core C++ compiler, linker, libs
    #   - ATL: required by llvm/include/llvm/DebugInfo/PDB/DIA/DIASupport.h (<atlbase.h>)
    #   - DIA SDK: PDB debug info reader, checked by cmake/config-ix.cmake (LLVM_ENABLE_DIA_SDK)
    #     The DIA SDK is part of the "Visual Studio C++ core features" component.
    #
    # Component IDs reference:
    #   https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools

    $requiredComponents = @(
        "Microsoft.VisualStudio.Workload.VCTools"
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
        "Microsoft.VisualStudio.Component.VC.Tools.ARM64"
        "Microsoft.VisualStudio.Component.VC.ATL"
        "Microsoft.VisualStudio.Component.VC.ATL.ARM64"
        "Microsoft.VisualStudio.Component.VC.DiagnosticTools"  # includes DIA SDK
        # The latest Windows SDK is included automatically via --includeRecommended
    )

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -nologo -latest -products '*' -format json | ConvertFrom-Json
        if ($vs) {
            Write-SubStep "Visual Studio already installed: $($vs[0].installationPath)"
            Write-SubStep "Ensuring required components are installed..."
            # setup.exe modify does not accept --wait; wait for the process
            # from PowerShell instead.
            $addArgs = @()
            foreach ($component in $requiredComponents) {
                $addArgs += '--add'
                $addArgs += $component
            }
            $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
            if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
                throw "Visual Studio Installer not found: $installer"
            }
            $installPath = $vs[0].installationPath
            $installArgs = @('modify', '--installPath', "`"$installPath`"") +
                $addArgs + @('--includeRecommended', '--passive', '--norestart')
            $process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
            if ($process.ExitCode -eq 3010) {
                throw 'Visual Studio components were installed, but Windows must be restarted before building.'
            }
            if ($process.ExitCode -ne 0) {
                throw "Failed to modify Visual Studio installation (exit code $($process.ExitCode))."
            }
            return
        }
    }

    Write-SubStep "Installing Visual Studio Build Tools 2022..."
    $addArgs = ($requiredComponents | ForEach-Object { "--add $_" }) -join ' '
    & $WingetPath install Microsoft.VisualStudio.2022.BuildTools `
        --accept-package-agreements --accept-source-agreements --silent `
        --override "$addArgs --includeRecommended --passive --wait"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Visual Studio Build Tools."
    }
}

#===============================================================================
# Visual Studio detection
#===============================================================================

function Find-VisualStudio {
    <#
    .SYNOPSIS
        Finds the Visual Studio installation and returns the path to VsDevCmd.bat.
    #>

    if ($env:VSINSTALLDIR) {
        Write-SubStep "Using enabled Visual Studio installation: $env:VSINSTALLDIR"
        $vsInstall = $env:VSINSTALLDIR
    } else {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) {
            throw "Cannot find vswhere.exe. Is Visual Studio installed?"
        }
        $vsInstall = & $vswhere -nologo -latest -products '*' -all -property installationPath
        if (-not $vsInstall) {
            throw "Cannot find any Visual Studio installation."
        }
        Write-SubStep "Detected Visual Studio: $vsInstall"
    }

    $vsDevCmd = Join-Path $vsInstall 'Common7' 'Tools' 'VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        throw "Cannot find VsDevCmd.bat at: $vsDevCmd"
    }
    return $vsDevCmd
}

function Enter-VsDevEnvironment {
    <#
    .SYNOPSIS
        Sources VsDevCmd.bat and imports the resulting environment variables into PowerShell.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$VsDevCmd,
        [Parameter(Mandatory)]
        [string]$Arch
    )
    Write-SubStep "Setting up VS developer environment for $Arch..."
    # Run VsDevCmd.bat in a cmd subprocess and capture the resulting environment
    $envBlock = cmd /c "`"$VsDevCmd`" -arch=$Arch -no_logo && set" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed for $Arch (exit code $LASTEXITCODE)."
    }
    # Reimport the environment variables from the previous command into the current shell.
    foreach ($line in $envBlock) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

function Use-DiaSdkRuntime {
    param([Parameter(Mandatory)][string]$Arch)

    if (-not $env:VSINSTALLDIR) {
        throw 'VsDevCmd did not set VSINSTALLDIR; cannot locate the DIA SDK runtime.'
    }
    # LLVM loads msdia140.dll by name when COM registration is unavailable.
    # Put the target architecture's DLL before other copies on PATH so an x86
    # DLL cannot be loaded by the x64 or arm64 test tools.
    $diaBin = Join-Path $env:VSINSTALLDIR "DIA SDK\bin\$Arch"
    $diaDll = Join-Path $diaBin 'msdia140.dll'
    if (-not (Test-Path -LiteralPath $diaDll -PathType Leaf)) {
        throw "DIA runtime not found: $diaDll. Install the Visual Studio Diagnostic Tools component."
    }
    $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $diaBin })
    $env:PATH = (@($diaBin) + $otherPaths) -join ';'
    Write-SubStep "Using DIA runtime: $diaDll"
}

#===============================================================================
# Python setup
#===============================================================================

function Test-PythonPsutil {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    try {
        # Lit clears PYTHONPATH and disables the user site. -I performs the
        # same checks and also avoids accidentally importing from the source tree.
        & $PythonExecutable -I -c 'import psutil' *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Install-PythonPsutil {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    if (Test-PythonPsutil -PythonExecutable $PythonExecutable) {
        Write-SubStep "Python psutil is already installed: $PythonExecutable"
        return
    }

    Write-SubStep "Installing Python psutil into $PythonExecutable..."
    & $PythonExecutable -I -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $PythonExecutable -I -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Could not bootstrap pip for $PythonExecutable."
        }
    }
    & $PythonExecutable -I -m pip install --disable-pip-version-check --no-input psutil
    if ($LASTEXITCODE -ne 0 -or -not (Test-PythonPsutil -PythonExecutable $PythonExecutable)) {
        throw "Could not install psutil into $PythonExecutable. Lit requires it for test timeouts."
    }
    Write-SubStep "Python psutil verified: $PythonExecutable"
}

function Install-PythonPsutilForRequestedArchitectures {
    $hostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if (-not $hostPython) {
        throw 'Python was installed, but its executable could not be found to install psutil.'
    }
    $pythonExecutables = @($hostPython)
    if ($arm64) {
        $arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($arm64Python) { $pythonExecutables += $arm64Python }
    }
    foreach ($pythonExecutable in ($pythonExecutables | Select-Object -Unique)) {
        Install-PythonPsutil -PythonExecutable $pythonExecutable
    }
}

function Get-PythonInstallDir {
    param([Parameter(Mandatory)][string]$PythonCommand)

    # Python Manager and winget can put only a shim in PATH. CMake needs the
    # actual installation directory, which sys.exec_prefix reports.
    $prefix = & $PythonCommand -c 'import sys; print(sys.exec_prefix)'
    if ($LASTEXITCODE -ne 0 -or -not $prefix) {
        throw "Could not find the Python installation behind $PythonCommand"
    }
    $installDir = ([string]$prefix).Trim()
    if (-not (Test-Path (Join-Path $installDir 'python.exe'))) {
        throw "Python reported $installDir, but python.exe is missing there."
    }
    return $installDir
}

function Test-PythonIsArm64 {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    try {
        $machine = & $PythonExecutable -c 'import platform; print(platform.machine())'
        return $LASTEXITCODE -eq 0 -and ([string]$machine).Trim() -eq 'ARM64'
    } catch {
        return $false
    }
}

function Get-PythonExecutableForArch {
    param([Parameter(Mandatory)][ValidateSet('amd64', 'arm64')][string]$Arch)

    if ($Arch -eq 'amd64') {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python -or -not (Test-ExecutableVersion -Path $python.Source)) {
            return $null
        }
        try {
            $pythonHome = Get-PythonInstallDir -PythonCommand $python.Source
            return Join-Path $pythonHome 'python.exe'
        } catch {
            return $null
        }
    }

    $basePath = "$env:LOCALAPPDATA\Programs\Python"
    if (Test-Path $basePath) {
        $candidates = @(Get-ChildItem -Path $basePath -Directory -Filter 'Python3*-arm64' |
            Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') } |
            Sort-Object Name -Descending)
        if ($candidates.Count -gt 0) {
            return Join-Path $candidates[0].FullName 'python.exe'
        }
    }
    return $null
}

function Find-Python {
    <#
    .SYNOPSIS
        Locates a Python installation for the given target architecture.

        For x64 (amd64) builds, uses the system Python from PATH.
        For ARM64 builds, probes standard install locations for any ARM64
        Python 3.x. Falls back to the PATH Python with a warning if no
        ARM64 install is found.
    #>
    param(
        [Parameter(Mandatory)][string]$Arch
    )

    $pythonHome = $null

    if ($Arch -eq 'amd64') {
        # Host-arch build: use system Python from PATH.
        $pythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
        if (-not $pythonExecutable) {
            throw "Cannot find python in PATH. Run with -InstallPrerequisites or install Python manually."
        }
        $pythonHome = Split-Path -Parent $pythonExecutable
    } else {
        # Cross-arch build: probe standard per-arch install locations.
        # Python installs to %LOCALAPPDATA%\Programs\Python\Python3XX-arm64
        $suffix = '-arm64'
        $basePath = "$env:LOCALAPPDATA\Programs\Python"
        $pythonExecutable = Get-PythonExecutableForArch -Arch 'arm64'
        if ($pythonExecutable) { $pythonHome = Split-Path -Parent $pythonExecutable }

        if (-not $pythonHome) {
            # Fall back to PATH Python with a warning.
            $pythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
            if ($pythonExecutable) {
                $pythonHome = Split-Path -Parent $pythonExecutable
                # On a native ARM64 machine (e.g. CI) the PATH Python is
                # already the right one.
                if (Test-PythonIsArm64 -PythonExecutable $pythonExecutable) {
                    Write-SubStep "PATH Python is already ARM64: $pythonHome"
                } else {
                    Write-Warning ("No per-arch Python found for $Arch (looked in $basePath\Python3*$suffix).`n" +
                        "  Falling back to PATH Python: $pythonHome`n" +
                        "  LLDB may not work correctly for the $Arch target.`n" +
                        "  Run with -InstallPrerequisites -$Arch to install the correct Python.")
                }
            } else {
                throw ("Cannot find Python for $Arch. No per-arch install in $basePath and no python in PATH.`n" +
                    "Run with -InstallPrerequisites to install Python.")
            }
        }
    }

    # Display the version without returning it alongside the Python directory.
    Invoke-NativeCommand (Join-Path $pythonHome 'python.exe') --version | Out-Host
    $env:PYTHONHOME = $pythonHome
    $env:PATH = "$pythonHome;$env:PATH"
    Write-SubStep "Using Python from: $pythonHome"
    return $pythonHome
}

#===============================================================================
# SWIG setup
#===============================================================================

function Find-Swig {
    <#
    .SYNOPSIS
        Locates SWIG's real executable and library directory.
    #>
    $command = Get-Command swig.exe -CommandType Application -ErrorAction Stop
    $executable = $command.Source
    $item = Get-Item -LiteralPath $executable -ErrorAction Stop

    # WinGet can put a symlink to swig.exe in PATH. SWIG may look for its
    # library beside that link rather than beside the installed executable.
    $linkType = $item.PSObject.Properties['LinkType']
    if ($linkType -and $linkType.Value -eq 'SymbolicLink') {
        $targetProperty = $item.PSObject.Properties['Target']
        $target = if ($targetProperty) { $targetProperty.Value } else { $null }
        if ($target -is [array]) { $target = $target[0] }
        if (-not $target) {
            throw "Cannot resolve SWIG symlink: $executable"
        }
        if (-not [IO.Path]::IsPathRooted($target)) {
            $target = Join-Path $item.DirectoryName $target
        }
        $executable = (Resolve-Path -LiteralPath $target -ErrorAction Stop).Path
    }

    $reportedLibDir = Invoke-NativeCommand $executable -swiglib
    if ($reportedLibDir -is [array]) { $reportedLibDir = $reportedLibDir[-1] }
    $libDir = ([string]$reportedLibDir).Trim()
    if (-not $libDir -or -not (Test-Path -LiteralPath (Join-Path $libDir 'swig.swg') -PathType Leaf)) {
        $libDir = Join-Path (Split-Path -Parent $executable) 'Lib'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $libDir 'swig.swg') -PathType Leaf)) {
        throw "SWIG library swig.swg was not found for $executable. Check the SWIG installation."
    }

    Write-SubStep "Using SWIG: $executable (library: $libDir)"
    return @{ Executable = $executable; LibraryDirectory = $libDir }
}

#===============================================================================
# Build helpers
#===============================================================================

# Suppress warnings in the downloaded dependencies with -w; clang, clang-cl,
# and cl all accept it.
function Build-LibXml2 {
    <#
    .SYNOPSIS
        Builds libxml2 and sets $script:LibXmlInstallDir to the install path.
    .NOTES
        Uses a script-scoped variable instead of return to avoid PowerShell
        capturing native command stdout as part of the function's return value,
        which would break Ninja's \r progress display and corrupt the path.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$SourceDir,
        [Parameter(Mandatory)]
        [string[]]$ToolchainFlags
    )
    Write-SubStep "Building libxml2..."
    $libxmlBuild = 'libxmlbuild'
    New-Item -ItemType Directory -Path $libxmlBuild -Force | Out-Null
    Push-Location $libxmlBuild
    try {
        $libxmlFlags = $ToolchainFlags + @(
            "CMAKE_BUILD_TYPE=Release"
            "CMAKE_C_FLAGS=-w"
            "CMAKE_INSTALL_PREFIX=$(Join-Path $PWD 'install')"
            "BUILD_SHARED_LIBS=OFF"
            "LIBXML2_WITH_C14N=OFF"
            "LIBXML2_WITH_CATALOG=OFF"
            "LIBXML2_WITH_DEBUG=OFF"
            "LIBXML2_WITH_DOCB=OFF"
            "LIBXML2_WITH_FTP=OFF"
            "LIBXML2_WITH_HTML=OFF"
            "LIBXML2_WITH_HTTP=OFF"
            "LIBXML2_WITH_ICONV=OFF"
            "LIBXML2_WITH_ICU=OFF"
            "LIBXML2_WITH_ISO8859X=OFF"
            "LIBXML2_WITH_LEGACY=OFF"
            "LIBXML2_WITH_LZMA=OFF"
            "LIBXML2_WITH_MEM_DEBUG=OFF"
            "LIBXML2_WITH_MODULES=OFF"
            "LIBXML2_WITH_OUTPUT=ON"
            "LIBXML2_WITH_PATTERN=OFF"
            "LIBXML2_WITH_PROGRAMS=OFF"
            "LIBXML2_WITH_PUSH=OFF"
            "LIBXML2_WITH_PYTHON=OFF"
            "LIBXML2_WITH_READER=OFF"
            "LIBXML2_WITH_REGEXPS=OFF"
            "LIBXML2_WITH_RUN_DEBUG=OFF"
            "LIBXML2_WITH_SAX1=ON"
            "LIBXML2_WITH_SCHEMAS=OFF"
            "LIBXML2_WITH_SCHEMATRON=OFF"
            "LIBXML2_WITH_TESTS=OFF"
            "LIBXML2_WITH_THREADS=ON"
            "LIBXML2_WITH_THREAD_ALLOC=OFF"
            "LIBXML2_WITH_TREE=ON"
            "LIBXML2_WITH_VALID=OFF"
            "LIBXML2_WITH_WRITER=OFF"
            "LIBXML2_WITH_XINCLUDE=OFF"
            "LIBXML2_WITH_XPATH=OFF"
            "LIBXML2_WITH_XPTR=OFF"
            "LIBXML2_WITH_ZLIB=OFF"
            "CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        )
        $cache = Write-CMakeCacheFile -Flags $libxmlFlags -FileName 'libxml2_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags $SourceDir
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install
        $script:LibXmlInstallDir = Get-ForwardSlashPath (Join-Path $PWD 'install')
    } finally {
        Pop-Location
    }
}

function Build-Zlib {
    <#
    .SYNOPSIS
        Builds zlib and sets $script:ZlibInstallDir to the install path.
    .NOTES
        See Build-LibXml2 for why a script-scoped variable is used instead of return.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$SourceDir,
        [Parameter(Mandatory)]
        [string[]]$ToolchainFlags
    )
    Write-SubStep "Building zlib..."
    $zlibBuild = 'zlibbuild'
    New-Item -ItemType Directory -Path $zlibBuild -Force | Out-Null
    Push-Location $zlibBuild
    try {
        $zlibFlags = $ToolchainFlags + @(
            "CMAKE_BUILD_TYPE=Release"
            "CMAKE_C_FLAGS=-w"
            "CMAKE_INSTALL_PREFIX=$(Join-Path $PWD 'install')"
            "ZLIB_BUILD_TESTING=OFF"
            "ZLIB_BUILD_SHARED=OFF"
            "ZLIB_BUILD_STATIC=ON"
            "ZLIB_INSTALL=ON"
            "CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        )
        $cache = Write-CMakeCacheFile -Flags $zlibFlags -FileName 'zlib_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags $SourceDir
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install
        $script:ZlibInstallDir = Get-ForwardSlashPath (Join-Path $PWD 'install')
    } finally {
        Pop-Location
    }
}

function Build-Zstd {
    <#
    .SYNOPSIS
        Builds zstd and sets $script:ZstdInstallDir to the install path.
    .NOTES
        See Build-LibXml2 for why a script-scoped variable is used instead of return.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$SourceDir,
        [Parameter(Mandatory)]
        [string[]]$ToolchainFlags
    )
    Write-SubStep "Building zstd..."
    $zstdBuild = 'zstdbuild'
    New-Item -ItemType Directory -Path $zstdBuild -Force | Out-Null
    Push-Location $zstdBuild
    try {
        $zstdFlags = $ToolchainFlags + @(
            "CMAKE_BUILD_TYPE=Release"
            "CMAKE_C_FLAGS=-w"
            "CMAKE_CXX_FLAGS=-w"
            "CMAKE_INSTALL_PREFIX=$(Join-Path $PWD 'install')"
            "ZSTD_BUILD_PROGRAMS=ON"
            "ZSTD_BUILD_TESTS=OFF"
            "ZSTD_BUILD_STATIC=ON"
            "ZSTD_BUILD_SHARED=OFF"
            # zstd's own CMakeLists.txt declares cmake_minimum_required(VERSION 3.10),
            # below the 3.15 that introduced CMP0091 -- so it defaults to OLD, and
            # CMAKE_MSVC_RUNTIME_LIBRARY below is silently ignored, leaving zstd on
            # CMake's traditional default (/MD, dynamic CRT). That mismatches the
            # rest of this build, which LLVM_ENABLE_RPMALLOC forces to /MT (see
            # LLVM_ENABLE_RPMALLOC's handling in llvm/CMakeLists.txt) -- producing
            # LNK4217 warnings ("locally defined symbol imported") for malloc/
            # calloc/free/etc. when zstd_static.lib links into llvm.exe. Forcing the
            # policy default (independent of zstd's own cmake_minimum_required)
            # makes CMAKE_MSVC_RUNTIME_LIBRARY actually take effect.
            "CMAKE_POLICY_DEFAULT_CMP0091=NEW"
            "CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        )
        $cache = Write-CMakeCacheFile -Flags $zstdFlags -FileName 'zstd_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$SourceDir/build/cmake"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install
        $script:ZstdInstallDir = Get-ForwardSlashPath (Join-Path $PWD 'install')
    } finally {
        Pop-Location
    }
}

function New-PGOProfile {
    <#
    .SYNOPSIS
        Builds and trains an instrumented stage 2 clang and sets the profile path.
    .NOTES
        Uses a script-scoped variable instead of return to avoid PowerShell
        capturing native command stdout as part of the function's return value,
        which would break Ninja's \r progress display and corrupt the path.

        PGO artifacts are created as top-level siblings of the stage 2 build
        directory so that each step owns an independent directory tree:
          $InstrumentDir/   - instrumented stage 2 clang build
          $TrainDir/        - training build
          $ProfilePath      - merged profile data
    #>
    param(
        [string[]]$CMakeFlags,
        [string]$Stage1BinDir,
        [string]$LlvmSrc,
        [Parameter(Mandatory)][string]$InstrumentDir,
        [Parameter(Mandatory)][string]$TrainDir,
        [Parameter(Mandatory)][string]$ProfilePath
    )
    Write-Step "Generating PGO profile for stage 2 self-hosted compiler"

    # Build instrumented stage 2 Clang
    Write-SubStep "Building instrumented stage 2 Clang..."
    New-Item -ItemType Directory -Path $InstrumentDir -Force | Out-Null
    Push-Location $InstrumentDir
    try {
        # The instrumented build only needs to produce an instrumented clang;
        # strip runtimes/projects that are unnecessary and would themselves be
        # compiled with -fprofile-generate (requiring the profile runtime to link).
        $instrumentFlags = $CMakeFlags + @(
            "LLVM_TARGETS_TO_BUILD=Native"
            "LLVM_BUILD_INSTRUMENTED=IR"
            'LLVM_ENABLE_RUNTIMES=""'
            'LLVM_ENABLE_PROJECTS="clang;lld"'
        )
        $cache = Write-CMakeCacheFile -Flags $instrumentFlags -FileName 'instrument_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs clang
        $instrumentedClang = Get-ForwardSlashPath (Join-Path $PWD 'bin' 'clang-cl.exe')
    } finally {
        Pop-Location
    }

    # Train: build part of LLVM with the instrumented stage 2 compiler
    Write-SubStep "Training with instrumented stage 2 Clang..."
    New-Item -ItemType Directory -Path $TrainDir -Force | Out-Null
    Push-Location $TrainDir
    try {
        $trainFlags = $CMakeFlags + @(
            "CMAKE_C_COMPILER=$instrumentedClang"
            "CMAKE_CXX_COMPILER=$instrumentedClang"
            "LLVM_ENABLE_PROJECTS=clang"
            'LLVM_ENABLE_RUNTIMES=""'
            "LLVM_TARGETS_TO_BUILD=Native"
        )
        $cache = Write-CMakeCacheFile -Flags $trainFlags -FileName 'train_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"

        # Drop profiles generated from running cmake; those are not representative.
        Remove-Item -Path "$InstrumentDir/profiles/*.profraw" -Force -ErrorAction SilentlyContinue
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs 'tools/clang/lib/Sema/CMakeFiles/obj.clangSema.dir/Sema.cpp.obj'
    } finally {
        Pop-Location
    }

    # Merge profiles
    $resolvedProfilePath = Get-ForwardSlashPath $ProfilePath
    Invoke-NativeCommand "$Stage1BinDir/llvm-profdata" merge `
        -output="$resolvedProfilePath" "$InstrumentDir/profiles/*.profraw"

    Write-SubStep "PGO profile generated: $resolvedProfilePath"
    $script:PGOProfilePath = $resolvedProfilePath
}

function Invoke-Build {
    param([string[]]$Targets = @())
    Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs @Targets
}

function Invoke-Tests {
    <#
    .SYNOPSIS
        Runs a list of test targets once each, skipping as appropriate.
        Honors $script:NinjaCommand and $script:NinjaExtraArgs (set via
        LLVM_NINJA_OVERRIDE).
    #>
    param(
        [string[]]$Targets,
        [string]$Arch = ''
    )
    foreach ($target in $Targets) {
        # Skip runtime checks on non-amd64
        if ($target -eq 'check-runtimes' -and $Arch -ne 'amd64') {
            Write-SubStep "Skipping $target on $Arch"
            continue
        }
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs $target
    }
}

function Assert-MsiUpgradeCode {
    param([Parameter(Mandatory)][string]$BuildDirectory)

    # This permanent UpgradeCode must match llvm/CMakeLists.txt.
    $expected = 'B08613CD-8BD0-4FB6-8937-621936604DE3'
    $msiFiles = @(Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.msi' -File)
    if ($msiFiles.Count -ne 1) {
        throw "Expected one MSI in $BuildDirectory, found $($msiFiles.Count)."
    }

    # Query Windows Installer directly; dark.exe is absent from newer WiX.
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $installer.OpenDatabase($msiFiles[0].FullName, 0)
    $view = $database.OpenView("SELECT Value FROM Property WHERE Property='UpgradeCode'")
    $view.Execute()
    $record = $view.Fetch()
    if (-not $record) {
        throw "Could not read the UpgradeCode from $($msiFiles[0].FullName)."
    }
    $actual = $record.StringData(1).Trim([char[]]'{}')
    if ($actual -ine $expected) {
        throw "Unexpected MSI UpgradeCode $actual in $($msiFiles[0].FullName); expected $expected."
    }
    Write-SubStep "Verified MSI UpgradeCode: $expected"
}

function Assert-MsiFileDeduplication {
    param([Parameter(Mandatory)][string]$BuildDirectory)

    $msiFiles = @(Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.msi' -File)
    if ($msiFiles.Count -ne 1) {
        throw "Expected one MSI in $BuildDirectory, found $($msiFiles.Count)."
    }

    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $installer.OpenDatabase($msiFiles[0].FullName, 0)
    $view = $database.OpenView('SELECT DestName FROM DuplicateFile')
    $view.Execute()
    $copies = @()
    while ($record = $view.Fetch()) {
        $copies += $record.StringData(1)
    }
    $view.Close()

    # Clang and LLD create these aliases through llvm_install_symlink. The
    # release MSI must create them from one payload instead of storing copies.
    foreach ($alias in @('clang-cl.exe', 'lld-link.exe')) {
        if ($alias -notin $copies) {
            throw "MSI is missing the DuplicateFile entry for $alias."
        }
    }
    Write-SubStep "Verified MSI file deduplication: $($copies.Count) copied filenames"
}

#===============================================================================
# Main build stages
#===============================================================================

function Build-Stage {
    <#
    .SYNOPSIS
        Builds the stage 1 bootstrap or stage 2 self-hosted compiler.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$Description,
        [Parameter(Mandatory)]
        [string[]]$CMakeFlags,
        [Parameter(Mandatory)]
        [string]$LlvmSrc,
        [string[]]$ExtraCMakeFlags = @(),
        [string[]]$InitialBuildTargets = @(),
        [switch]$SkipFullBuild,
        [string[]]$TestTargets = @(),
        [string]$Arch = ''
    )

    Write-Step "Building $Description ($Name)"
    New-Item -ItemType Directory -Path $Name -Force | Out-Null
    Push-Location $Name
    try {
        $allFlags = $CMakeFlags + $ExtraCMakeFlags
        $cache = Write-CMakeCacheFile -Flags $allFlags
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
        if ($InitialBuildTargets.Count -gt 0) {
            Invoke-Build -Targets $InitialBuildTargets
        }
        if (-not $SkipFullBuild) {
            Invoke-Build
            if ($TestTargets.Count -gt 0) {
                Invoke-Tests -Targets $TestTargets -Arch $Arch
            }
        }
    } finally {
        Pop-Location
    }
}

function Build-Architecture {
    <#
    .SYNOPSIS
        Selects the stage 0 host toolchain, then builds the stage 1 bootstrap
        and stage 2 self-hosted compilers for an architecture.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Arch,
        [Parameter(Mandatory)]
        [string]$VsDevCmd,
        [Parameter(Mandatory)]
        [string]$LlvmSrc,
        [Parameter(Mandatory)]
        [string]$BuildDir,
        [Parameter(Mandatory)]
        [string]$ThirdPartyDir,
        [Parameter(Mandatory)]
        [string]$PackageVersion,
        [Parameter(Mandatory)]
        [string[]]$CommonCMakeFlags,
        [Parameter(Mandatory)]
        [string]$CommonCompilerFlags,
        [Parameter(Mandatory)]
        [string[]]$Stage2CMakeFlags,
        [string[]]$CommonLLDBFlags = @(),
        [switch]$UseFastBuild
    )

    Write-Step "Building for architecture: $Arch"

    # Restore clean PATH and setup environment
    $env:PATH = $script:OriginalPath
    $pythonHome = Find-Python -Arch $Arch
    $pythonExecutable = Join-Path $pythonHome 'python.exe'
    if (-not (Test-PythonPsutil -PythonExecutable $pythonExecutable)) {
        throw "Python psutil is missing from $pythonExecutable. Run with -InstallPrerequisites to enable lit test timeouts."
    }
    Enter-VsDevEnvironment -VsDevCmd $VsDevCmd -Arch $Arch
    Use-DiaSdkRuntime -Arch $Arch

    # Stage 0 uses the official LLVM release unless MSVC was requested.
    $hostClangCl = $null
    $hostLldLink = $null
    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if (-not $llvmBin) {
            throw 'Official LLVM clang-cl.exe and lld-link.exe are missing. Run with -InstallPrerequisites, or use -ForceMSVC.'
        }
        $env:PATH = "$llvmBin;$env:PATH"
        $hostClangCl = Join-Path $llvmBin 'clang-cl.exe'
        $hostLldLink = Join-Path $llvmBin 'lld-link.exe'
    }

    if ($hostClangCl) {
        $hostCompiler = $hostClangCl
        $hostLinker = $hostLldLink
        $CommonCompilerFlags += ' -fuse-ld=lld'
        $CommonCMakeFlags += @(
            "LLVM_ENABLE_LLD=ON"
            "CMAKE_C_FLAGS=`"$CommonCompilerFlags`""
            "CMAKE_CXX_FLAGS=`"$CommonCompilerFlags`""
        )
    } else {
        # Pin the MSVC driver so bcrypt.lib is interpreted as a library.
        $hostCompiler = 'cl.exe'
        $hostLinker = 'link.exe'
    }
    Write-SubStep "Stage 0 host compiler: $hostCompiler (linker: $hostLinker)"

    $hostToolchainFlags = @(
        "CMAKE_C_COMPILER=$hostCompiler"
        "CMAKE_CXX_COMPILER=$hostCompiler"
        "CMAKE_LINKER=$hostLinker"
    )
    $CommonCMakeFlags += $hostToolchainFlags

    $env:VSCMD_START_DIR = $BuildDir
    $swigConfig = Find-Swig
    # CMake and Ninja subprocesses need the library when invoking SWIG.
    $env:SWIG_LIB = $swigConfig.LibraryDirectory

    # Directory names (always computed, used by multiple steps)
    $stage1Name = "build_${Arch}_stage1"
    $stage2Name = "build_${Arch}_stage2"
    $instrumentDir = Join-Path $BuildDir "instrument_${Arch}_stage2"
    $trainDir      = Join-Path $BuildDir "train_${Arch}_stage2"
    $profileFile   = Join-Path $BuildDir "profile_${Arch}_stage2.profdata"

    #-------------------------------------------------------------------
    # Step: libxml2
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'libxml2') {
        # Every step that runs starts from a clean directory, so artifacts of
        # a previous run cannot leak into this one.
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'libxmlbuild')
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'zlibbuild')
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'zstdbuild')
        New-Item -ItemType Directory -Path $stage1Name -Force | Out-Null
        Push-Location $stage1Name
        try {
            Build-LibXml2 -SourceDir (Join-Path $ThirdPartyDir 'libxml2-v2.15.3') -ToolchainFlags $hostToolchainFlags
            Build-Zlib -SourceDir (Join-Path $ThirdPartyDir 'zlib-1.3.2') -ToolchainFlags $hostToolchainFlags
            Build-Zstd -SourceDir (Join-Path $ThirdPartyDir 'zstd-1.5.7') -ToolchainFlags $hostToolchainFlags
        } finally {
            Pop-Location
        }
    } else {
        Write-Step "Skipping libxml2 (-StartAt $($script:StartAtStep))"
        $libxmlInstallPath = Join-Path $BuildDir $stage1Name 'libxmlbuild' 'install'
        $zlibInstallPath = Join-Path $BuildDir $stage1Name 'zlibbuild' 'install'
        $zstdInstallPath = Join-Path $BuildDir $stage1Name 'zstdbuild' 'install'
        Assert-PathExists -Path $libxmlInstallPath -Description 'libxml2 install directory'
        Assert-PathExists -Path $zlibInstallPath -Description 'zlib install directory'
        Assert-PathExists -Path $zstdInstallPath -Description 'zstd install directory'
        $script:LibXmlInstallDir = Get-ForwardSlashPath $libxmlInstallPath
        $script:ZlibInstallDir = Get-ForwardSlashPath $zlibInstallPath
        $script:ZstdInstallDir = Get-ForwardSlashPath $zstdInstallPath
    }
    $libxmlDir = $script:LibXmlInstallDir
    $zlibDir = $script:ZlibInstallDir
    $zstdDir = $script:ZstdInstallDir

    # The stage 1 bootstrap tools are needed by PGO and stage 2.
    $stage1BinDir = Get-ForwardSlashPath (Join-Path $BuildDir $stage1Name 'bin')

    # Compute cmakeFlags (always needed, cheap -- derived from parameters)
    $cmakeFlags = $CommonCMakeFlags + @(
        "Python3_ROOT_DIR=$env:PYTHONHOME"
        "LIBXML2_INCLUDE_DIR=$libxmlDir/include/libxml2"
        "LIBXML2_LIBRARY=$libxmlDir/lib/libxml2s.lib"
        "LIBXML2_LIBRARIES=$libxmlDir/lib/libxml2s.lib;ws2_32.lib"
        "ZLIB_INCLUDE_DIR=$zlibDir/include"
        "ZLIB_LIBRARY=$zlibDir/lib/zs.lib"
        "ZLIB_LIBRARY_RELEASE=$zlibDir/lib/zs.lib"
        "zstd_INCLUDE_DIR=$zstdDir/include"
        "zstd_LIBRARY=$zstdDir/lib/zstd_static.lib"
    )

    $cmakeFlags += "CLANG_DEFAULT_LINKER=lld"
    if ($Arch -eq 'arm64') {
        $cmakeFlags += "COMPILER_RT_BUILD_SANITIZERS=OFF"
    }

    #-------------------------------------------------------------------
    # Step: stage 1 bootstrap compiler
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'stage1') {
        # Preserve the dependency installs needed by stage 1.
        $stage1Dir = Join-Path $BuildDir $stage1Name
        if (Test-Path $stage1Dir) {
            Write-SubStep "Cleaning stage 1 bootstrap build artifacts (preserving dependency installs)..."
            Get-ChildItem -Path $stage1Dir -Exclude 'libxmlbuild', 'zlibbuild', 'zstdbuild' |
                Remove-Item -Recurse -Force
        }

        # Stage 2 needs these tools even when the fast build skips the rest
        # of the stage 1 bootstrap build and its tests.
        $stage1InitialTargets = @('clang', 'lld', 'llvm-lib', 'llvm-rc', 'runtimes')
        if ($Arch -eq 'amd64') {
            $stage1InitialTargets += 'llvm-ml64'
        }
        $skipStage1FullBuild = $UseFastBuild
        $stage1Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes')
        $stage1Extra = @("LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON",
                         "LLVM_TARGETS_TO_BUILD=Native")

        Build-Stage -Name $stage1Name -Description 'stage 1 bootstrap compiler' `
            -CMakeFlags $cmakeFlags `
            -LlvmSrc $LlvmSrc `
            -ExtraCMakeFlags $stage1Extra `
            -InitialBuildTargets $stage1InitialTargets `
            -SkipFullBuild:$skipStage1FullBuild `
            -TestTargets $stage1Tests `
            -Arch $Arch
    } else {
        Write-Step "Skipping stage 1 bootstrap compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path "$stage1BinDir/clang-cl.exe" -Description 'stage 1 bootstrap clang-cl.exe'
        Assert-PathExists -Path "$stage1BinDir/lld-link.exe" -Description 'stage 1 bootstrap lld-link.exe'
        Assert-PathExists -Path "$stage1BinDir/llvm-lib.exe" -Description 'stage 1 bootstrap llvm-lib.exe'
        Assert-PathExists -Path "$stage1BinDir/llvm-rc.exe" -Description 'stage 1 bootstrap llvm-rc.exe'
        if ($Arch -eq 'amd64') {
            Assert-PathExists -Path "$stage1BinDir/llvm-ml64.exe" -Description 'stage 1 bootstrap llvm-ml64.exe'
        }
        # -FastBuild does not build llvm-profdata, which the PGO merge needs.
        if (-not $UseFastBuild -and (Test-ShouldRun 'pgo')) {
            Assert-PathExists -Path "$stage1BinDir/llvm-profdata.exe" -Description 'stage 1 bootstrap llvm-profdata.exe (not built by -FastBuild)'
        }
    }

    # Stage 2 is built with the stage 1 bootstrap toolchain.
    $stage2CompilerFlags = @(
        "CMAKE_C_COMPILER=$stage1BinDir/clang-cl.exe"
        "CMAKE_CXX_COMPILER=$stage1BinDir/clang-cl.exe"
        "CMAKE_LINKER=$stage1BinDir/lld-link.exe"
        "CMAKE_AR=$stage1BinDir/llvm-lib.exe"
        "CMAKE_RC_COMPILER=$stage1BinDir/llvm-rc.exe"
    )

    if ($Arch -eq 'amd64') {
        $stage2CompilerFlags += "CMAKE_ASM_MASM_COMPILER=$stage1BinDir/llvm-ml64.exe"
    } else {
        $stage2CompilerFlags += "CPACK_SYSTEM_NAME=woa64"
    }

    $stage2Flags = $cmakeFlags + $stage2CompilerFlags

    #-------------------------------------------------------------------
    # Step: pgo
    #-------------------------------------------------------------------
    $profileFlags = @()
    if (-not $UseFastBuild) {
        if (Test-ShouldRun 'pgo') {
            Remove-StepDirectory $instrumentDir
            Remove-StepDirectory $trainDir
            Remove-StepDirectory $profileFile
            New-PGOProfile -CMakeFlags $stage2Flags `
                -Stage1BinDir $stage1BinDir -LlvmSrc $LlvmSrc `
                -InstrumentDir $instrumentDir `
                -TrainDir $trainDir `
                -ProfilePath $profileFile
        } else {
            Write-Step "Skipping PGO (-StartAt $($script:StartAtStep))"
            Assert-PathExists -Path $profileFile -Description 'PGO profile data'
            $script:PGOProfilePath = Get-ForwardSlashPath $profileFile
        }
        $profilePath = $script:PGOProfilePath

        $pgoCFlags = "$CommonCompilerFlags -Wno-backend-plugin"
        $profileFlags = @(
            "LLVM_PROFDATA_FILE=$profilePath"
            "CMAKE_C_FLAGS=`"$pgoCFlags`""
            "CMAKE_CXX_FLAGS=`"$pgoCFlags`""
        )
    }

    #-------------------------------------------------------------------
    # Step: stage 2 self-hosted compiler
    #-------------------------------------------------------------------
    $stage2Projects = '"clang;clang-tools-extra;lld;lldb;flang;mlir"'

    $stage2Extra = @(
        "LLVM_ENABLE_PROJECTS=$stage2Projects"
        'LLVM_ENABLE_RUNTIMES="compiler-rt;openmp"'
        "PYTHON_HOME=$env:PYTHONHOME"
    ) + $CommonLLDBFlags + $Stage2CMakeFlags + @(
        "SWIG_EXECUTABLE=$($swigConfig.Executable)"
        "SWIG_DIR=$($swigConfig.LibraryDirectory)"
    ) + $profileFlags

    if (Test-ShouldRun 'stage2') {
        # Also discards the settings the tarball step left in the CMake cache.
        Remove-StepDirectory (Join-Path $BuildDir $stage2Name)

        $stage2Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes',
                         'check-clang-tools', 'check-clangd')

        Build-Stage -Name $stage2Name -Description 'stage 2 self-hosted compiler' `
            -CMakeFlags $stage2Flags `
            -LlvmSrc $LlvmSrc `
            -ExtraCMakeFlags $stage2Extra `
            -TestTargets $stage2Tests `
            -Arch $Arch
    } else {
        Write-Step "Skipping stage 2 self-hosted compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path (Join-Path $BuildDir $stage2Name) -Description 'stage 2 build directory'
    }

    #-------------------------------------------------------------------
    # Step: package (WiX MSI installer)
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'package') {
        Write-Step "Creating WiX MSI installer package"
        Push-Location $stage2Name
        try {
            # The tarball step reconfigures this shared build directory with
            # LLVM_INSTALL_TOOLCHAIN_ONLY=OFF. Pin the stage 2 settings again
            # so a resumed package step still produces the MSI payload.
            $packageFlags = $stage2Flags + $stage2Extra
            $cache = Write-CMakeCacheFile -Flags $packageFlags -FileName 'package_cache.cmake'
            $otherFlags = $cache.OtherFlags
            Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"

            # CPack does not remove stale files from its staging tree, so
            # clean it after reconfiguring from a possible tarball build.
            Remove-StepDirectory (Join-Path $PWD '_CPack_Packages')

            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs package
            # Downloaded release tags can predate these MSI settings. Check
            # each invariant only when the source being packaged defines it.
            $llvmCMakeLists = Join-Path $LlvmSrc 'llvm/CMakeLists.txt'
            if (Select-String -LiteralPath $llvmCMakeLists -Pattern 'set\(CPACK_WIX_UPGRADE_GUID' -Quiet) {
                Assert-MsiUpgradeCode -BuildDirectory $PWD.Path
            }
            if (Select-String -LiteralPath $llvmCMakeLists -Pattern 'CPACK_WIX_DEDUPLICATION_PATCH_FILE' -Quiet) {
                Assert-MsiFileDeduplication -BuildDirectory $PWD.Path
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Step "Skipping package (-StartAt $($script:StartAtStep))"
    }

    #-------------------------------------------------------------------
    # Step: tarball
    #-------------------------------------------------------------------
    $tripleArch = if ($Arch -eq 'amd64') { 'x86_64' } else { 'aarch64' }
    $filename = "clang+llvm-${PackageVersion}-${tripleArch}-pc-windows-msvc"

    if (Test-ShouldRun 'tarball') {
        Remove-StepDirectory (Join-Path $BuildDir $filename)
        Remove-StepDirectory (Join-Path $BuildDir "$filename.tar")
        Remove-StepDirectory (Join-Path $BuildDir "$filename.tar.xz")

        Push-Location $stage2Name
        try {
            $tarballFlags = $stage2Flags + $stage2Extra + @(
                "LLVM_INSTALL_TOOLCHAIN_ONLY=OFF"
                "CMAKE_INSTALL_PREFIX=$BuildDir/$filename"
                "LLVM_INCLUDE_TESTS=OFF"
            )
            $cache = Write-CMakeCacheFile -Flags $tarballFlags -FileName 'tarball_cache.cmake'
            $otherFlags = $cache.OtherFlags
            Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install

            # Verify llvm-config works
            Invoke-NativeCommand "$BuildDir/$filename/bin/llvm-config.exe" --bindir
        } finally {
            Pop-Location
        }

        # Create the portable archive without changing executable layout.
        Invoke-NativeCommand 7z a -ttar "$filename.tar" $filename
        Invoke-NativeCommand 7z a -txz "$filename.tar.xz" "$filename.tar"
        Remove-Item -LiteralPath "$filename.tar"
    } else {
        Write-Step "Skipping tarball (-StartAt $($script:StartAtStep))"
    }
}

#===============================================================================
# Main
#===============================================================================

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

if ($PrerequisitesOnly -and -not $InstallPrerequisites) {
    throw '-PrerequisitesOnly requires -InstallPrerequisites.'
}

# Install prerequisites if requested
if ($InstallPrerequisites) {
    Install-Prerequisites
}
if ($PrerequisitesOnly) { exit 0 }

# Offer to install missing tools in interactive mode, then validate again.
$missing = @(Get-MissingPrerequisites)
if ($missing.Count -gt 0 -and -not $InstallPrerequisites -and -not $Unattended) {
    Write-Host "Missing prerequisites: $($missing -join ', ')" -ForegroundColor Yellow
    $answer = Read-Host "Install prerequisites (including NetFx3 and Visual Studio Build Tools if needed)? [y/N]"
    if ($answer -eq 'y' -or $answer -eq 'Y') {
        Install-Prerequisites
        $missing = @(Get-MissingPrerequisites)
    } else {
        Write-Host "Prerequisites were not installed. Aborted."
        exit 1
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Error ("Missing prerequisites: $($missing -join ', ')`n" +
        "Run the script with -InstallPrerequisites to install them, or install manually.")
    exit 1
}

if ($DownloadSource -and -not $StartAt) {
    Assert-SevenZipSymlinkSupport
}
Write-SubStep "All prerequisites found."

# Detect Visual Studio
$vsDevCmd = Find-VisualStudio

# Determine LLVM source directory
if ($DownloadSource) {
    $llvmSrc = $null  # Set after the build directory is known.
} else {
    $llvmSrc = Resolve-Path (Join-Path $PSScriptRoot '..\..\..') | Select-Object -ExpandProperty Path
}

# Auto-detect or validate version
if ($Version) {
    $packageVersion = $Version
    Write-Host "Using specified version: $packageVersion"
} else {
    if ($llvmSrc) {
        $versionInfo = Get-LLVMVersionFromSource -SourceDir $llvmSrc
        $packageVersion = $versionInfo.Full
        Write-Host "Auto-detected LLVM version from source: $packageVersion"
    } else {
        # -DownloadSource without -Version: try detecting from the repo the script lives in.
        $repoRoot = Join-Path $PSScriptRoot '..\..\..'
        if (Test-Path (Join-Path $repoRoot 'cmake' 'Modules' 'LLVMVersion.cmake')) {
            $versionInfo = Get-LLVMVersionFromSource -SourceDir (Resolve-Path $repoRoot)
            $packageVersion = $versionInfo.Full
            Write-Host "Auto-detected LLVM version from repo: $packageVersion"
        } else {
            Write-Error "Cannot auto-detect version. Use -Version <version> when using -DownloadSource."
            exit 1
        }
    }
}

$revision = "llvmorg-$packageVersion"
$buildDir = Join-Path $PWD "llvm_package_$packageVersion"
$thirdPartyDir = Join-Path $buildDir 'third_party'

Write-Step "Configuration"
Write-Host "  Revision:        $revision"
Write-Host "  Package version: $packageVersion"
Write-Host "  Build dir:       $buildDir"
Write-Host "  Architectures:   $((@('x64','arm64') | Where-Object { (Get-Variable $_ -ValueOnly) }) -join ', ')"

if ($StartAt) {
    # Resuming from a specific step: validate that the build directory exists.
    if (-not (Test-Path $buildDir)) {
        Write-Error "Build directory does not exist: $buildDir`nCannot use -StartAt without a previous build. Run a full build first."
        exit 1
    }
    Write-Host "  Resuming from step: $StartAt" -ForegroundColor Yellow
} else {
    # Full build: prompt to delete existing build directory.
    if (Test-Path $buildDir) {
        if ($Unattended) {
            Write-Error "Build directory already exists: $buildDir`nUse -StartAt to resume it, or remove it before running an unattended full build."
            exit 1
        }
        $answer = Read-Host "Build directory already exists: $buildDir`nDelete and re-create it? [y/N]"
        if ($answer -ne 'y' -and $answer -ne 'Y') {
            Write-Host "Aborted."
            exit 1
        }
        Remove-Item -Recurse -Force $buildDir
    }
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Push-Location $buildDir

# Ninja's single-line progress display overprints cmake's output and garbles
# the console on some hosts (notably conhost). TERM=dumb makes ninja print one
# plain line per step, which also keeps CI logs readable. Windows Terminal
# (identified by WT_SESSION) renders the progress line correctly, and an
# explicit TERM set by the caller is respected, so neither is overridden.
$script:SavedTerm = $env:TERM
if (-not $env:TERM -and -not $env:WT_SESSION) { $env:TERM = 'dumb' }

try {
    # Download source if requested (skip when resuming with -StartAt)
    if ($DownloadSource) {
        $llvmSrc = Join-Path $buildDir 'llvm-project'
    }
    if ($DownloadSource -and -not $StartAt) {
        Write-Step "Downloading $revision"
        Invoke-NativeCommand curl.exe -L `
            "https://github.com/llvm/llvm-project/archive/$revision.zip" -o src.zip
        Invoke-NativeCommand 7z x src.zip
        Get-ChildItem -Directory -Filter 'llvm-project-*' |
            Rename-Item -NewName 'llvm-project'
    }
    if ($DownloadSource) {
        Assert-PathExists -Path $llvmSrc -Description 'downloaded LLVM source directory'
    }

    # Keep third-party archives and extracted sources together.
    New-Item -ItemType Directory -Path $thirdPartyDir -Force | Out-Null
    Push-Location $thirdPartyDir
    try {
        # Download and extract libxml2 (skip if resuming and it already exists)
        if ($StartAt -and (Test-Path 'libxml2-v2.15.3')) {
            Write-Step "Skipping libxml2 download (already exists)"
        } else {
            Write-Step "Downloading libxml2"
            Invoke-NativeCommand curl.exe --remote-name `
                'https://gitlab.gnome.org/GNOME/libxml2/-/archive/v2.15.3/libxml2-v2.15.3.tar.gz'
            Test-FileChecksum -Path 'libxml2-v2.15.3.tar.gz' -Algorithm SHA256 `
                -ExpectedHash '0DA50C1415F4EC0364569D2119B1436BA837B31DF44AF28569D234272C23CF1F'
            # The test directory contains symlinks.
            Invoke-NativeCommand tar zxf 'libxml2-v2.15.3.tar.gz' --exclude 'test/*'
        }

        # Download and extract zlib (skip if resuming and it already exists)
        if ($StartAt -and (Test-Path 'zlib-1.3.2')) {
            Write-Step "Skipping zlib download (already exists)"
        } else {
            Write-Step "Downloading zlib"
            Invoke-NativeCommand curl.exe -LO `
                'https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz'
            Test-FileChecksum -Path 'zlib-1.3.2.tar.gz' -Algorithm SHA256 `
                -ExpectedHash 'BB329A0A2CD0274D05519D61C667C062E06990D72E125EE2DFA8DE64F0119D16'
            Invoke-NativeCommand tar zxf 'zlib-1.3.2.tar.gz'
        }

        # Download and extract zstd (skip if resuming and it already exists)
        if ($StartAt -and (Test-Path 'zstd-1.5.7')) {
            Write-Step "Skipping zstd download (already exists)"
        } else {
            Write-Step "Downloading zstd"
            Invoke-NativeCommand curl.exe -LO `
                'https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz'
            Test-FileChecksum -Path 'zstd-1.5.7.tar.gz' -Algorithm SHA256 `
                -ExpectedHash 'EB33E51F49A15E023950CD7825CA74A4A2B43DB8354825AC24FC1B7EE09E6FA3'
            # 'tests' directory excluded because of symlinks.
            Invoke-NativeCommand tar zxf 'zstd-1.5.7.tar.gz' --exclude 'tests/*'
        }
    } finally {
        Pop-Location
    }

    # Preserve original PATH
    $script:OriginalPath = $env:PATH

    # Common flags
    $commonCompilerFlags = '-DLIBXML_STATIC -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING'
    $releaseCompilerFlags = '/O2 /Ob2 /DNDEBUG'
    # -FastBuild skips ThinLTO to keep the build time down.
    $ltoFlags = if ($FastBuild) { '' } else { ' -flto=thin -fsplit-lto-unit' }
    $wpvFlags = if ($FastBuild) { '' } else { ' -fwhole-program-vtables' }
    $stage2CMakeFlags = @(
        # Pass the stage 1 bootstrap assembler to OpenMP's runtimes build too.
        'LLVM_EXTERNAL_PROJECT_PASSTHROUGH=CMAKE_ASM_MASM_COMPILER'
        # Early CMake probes need ThinLTO, and whole-program vtables require
        # every ThinLTO unit to use the same splitting mode.
        "CMAKE_C_FLAGS_RELEASE=$releaseCompilerFlags -fstrict-aliasing$ltoFlags"
        "CMAKE_CXX_FLAGS_RELEASE=$releaseCompilerFlags -Wno-unused-template -fstrict-aliasing$ltoFlags$wpvFlags"
    )
    if (-not $FastBuild) { $stage2CMakeFlags += 'LLVM_ENABLE_LTO=THIN' }
    $commonCMakeFlags = @(
        "CMAKE_BUILD_TYPE=Release"
        "LLVM_ENABLE_ASSERTIONS=OFF"
        "LLVM_INSTALL_TOOLCHAIN_ONLY=ON"
        # lit retries each failing test once without rerunning passing tests.
        "LLVM_LIT_ARGS=-sv --max-retries-per-test=1"
        'LLVM_TARGETS_TO_BUILD="AArch64;ARM;X86;BPF;WebAssembly;RISCV;NVPTX"'
        "LLVM_BUILD_LLVM_C_DYLIB=ON"
        "Python3_FIND_REGISTRY=NEVER"
        "PACKAGE_VERSION=$packageVersion"
        'CMAKE_CL_SHOWINCLUDES_PREFIX="Note: including file: "'
        "LLVM_ENABLE_LIBXML2=FORCE_ON"
        "CLANG_ENABLE_LIBXML2=OFF"
        "LLVM_ENABLE_ZLIB=FORCE_ON"
        "LLVM_ENABLE_ZSTD=FORCE_ON"
        "CMAKE_C_FLAGS=`"$commonCompilerFlags`""
        "CMAKE_CXX_FLAGS=`"$commonCompilerFlags`""
        "CMAKE_CXX_FLAGS_RELEASE=`"$releaseCompilerFlags`""
        # libxml2 2.15 needs bcrypt on executable and shared links.
        "CMAKE_EXE_LINKER_FLAGS=bcrypt.lib"
        "CMAKE_SHARED_LINKER_FLAGS=bcrypt.lib"
        "LLVM_ENABLE_RPMALLOC=ON"
        'LLVM_ENABLE_PROJECTS="clang;lld"'
        'LLVM_ENABLE_RUNTIMES="compiler-rt"'
        "COMPILER_RT_BUILD_ORC=OFF"
        "LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
        "CMAKE_DISABLE_PRECOMPILE_HEADERS=ON"
        # Configure WiX explicitly so llvm/CMakeLists.txt sets its MSI
        # compression, install scope, and permanent UpgradeCode.
        "CPACK_GENERATOR=WIX"
    )

    $commonLLDBFlags = @(
        "LLDB_RELOCATABLE_PYTHON=1"
        "LLDB_EMBED_PYTHON_HOME=OFF"
    )

    # Build each requested architecture
    $buildParams = @{
        VsDevCmd            = $vsDevCmd
        LlvmSrc             = $llvmSrc
        BuildDir            = $buildDir
        ThirdPartyDir       = $thirdPartyDir
        PackageVersion      = $packageVersion
        CommonCMakeFlags    = $commonCMakeFlags
        CommonCompilerFlags = $commonCompilerFlags
        Stage2CMakeFlags    = $stage2CMakeFlags
        CommonLLDBFlags     = $commonLLDBFlags
        UseFastBuild        = $FastBuild
    }

    if ($x64)   { Build-Architecture -Arch 'amd64'  @buildParams }
    if ($arm64) { Build-Architecture -Arch 'arm64'  @buildParams }

    Write-Step "Build complete!"
    Write-Host "Packages are in: $buildDir" -ForegroundColor Green

} finally {
    Pop-Location
    $env:TERM = $script:SavedTerm
    # Restore the console mode that was saved at script start.
    if ($null -ne $script:SavedConsoleMode) {
        try { [ConsoleMode]::Set($script:SavedConsoleMode) } catch {}
    }
}
