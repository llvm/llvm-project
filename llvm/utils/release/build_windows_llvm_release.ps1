#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script for building the LLVM installer on Windows,
    used for the releases at https://github.com/llvm/llvm-project/releases

.DESCRIPTION
    Builds LLVM release packages for Windows (x64, arm64).
    Performs a 2-stage build with a ThinLTO final stage and optional PGO.
    Can bootstrap a fresh VM by installing all prerequisites.

    Builds for the architecture reported by PROCESSOR_ARCHITECTURE, using the
    local source tree and auto-detected Python. Use -DownloadSource to download
    a tagged release tarball instead.

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
    On ARM64, also installs ARM64 Python for LLDB.
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
    .\build_windows_llvm_release.ps1
    Build for the current process architecture using the local source tree.

.EXAMPLE
    .\build_windows_llvm_release.ps1 -Version 19.1.0 -DownloadSource
    Download version 19.1.0 sources and build for the current process architecture.

.EXAMPLE
    .\build_windows_llvm_release.ps1 -InstallPrerequisites
    Install prerequisites, then build for the current process architecture.

.EXAMPLE
    .\build_windows_llvm_release.ps1 -Unattended -InstallPrerequisites
    From an elevated shell, install prerequisites and build without prompts.

.EXAMPLE
    .\build_windows_llvm_release.ps1 -StartAt stage2
    Resume from stage 2 (reuses the stage 1 bootstrap compiler and PGO profile).

.EXAMPLE
    .\build_windows_llvm_release.ps1 -StartAt package
    Resume at WiX MSI packaging, then regenerate the portable archive.

.EXAMPLE
    .\build_windows_llvm_release.ps1 -StartAt ""
    Display the list of available build steps and exit.

.NOTES
    PROCESSOR_ARCHITECTURE must be AMD64 or ARM64. Python is auto-detected from
    PATH for AMD64 builds. For ARM64 builds,
    the script probes standard install locations for ARM64 Python. Use
    -InstallPrerequisites to install it automatically.

    Environment variables:
      PROCESSOR_ARCHITECTURE - Selects the build architecture (AMD64 or ARM64).
      LLVM_NINJA_OVERRIDE  - Override the ninja binary and optionally provide
                             extra flags. The first token is the executable,
                             remaining tokens are prepended to every ninja
                             invocation.
                             Example: LLVM_NINJA_OVERRIDE="myninja.exe --flag1 --flag2"
#>

param(
    [string]$Version,
    [switch]$DownloadSource,
    [switch]$ForceMSVC,
    [switch]$FastBuild,
    [switch]$InstallPrerequisites,
    [switch]$Unattended,
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
$script:ReleaseScriptRoot = $PSScriptRoot
$script:MinimumCMakeVersion = [version]'3.31.0'
$script:BuildArch = switch ($env:PROCESSOR_ARCHITECTURE) {
    'AMD64' { 'amd64' }
    'ARM64' { 'arm64' }
    default { throw "Unsupported PROCESSOR_ARCHITECTURE: '$env:PROCESSOR_ARCHITECTURE'. Expected AMD64 or ARM64." }
}

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
#===============================================================================
# Shared release helpers
#===============================================================================
# Keep these dot-sourced so shared $script: state and functions stay in this scope.
$windowsScriptDir = Join-Path $script:ReleaseScriptRoot 'windows'
. (Join-Path $windowsScriptDir 'Common.ps1')
. (Join-Path $windowsScriptDir 'Prerequisites.ps1')
. (Join-Path $windowsScriptDir 'Toolchain.ps1')
. (Join-Path $windowsScriptDir 'ThirdParty.ps1')
. (Join-Path $windowsScriptDir 'Package.ps1')
. (Join-Path $windowsScriptDir 'Build.ps1')

#===============================================================================
# Main
#===============================================================================

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Install prerequisites if requested
if ($InstallPrerequisites) {
    Install-Prerequisites
}

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
    $llvmSrc = Resolve-Path (Join-Path $script:ReleaseScriptRoot '..\..\..') | Select-Object -ExpandProperty Path
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
        $repoRoot = Join-Path $script:ReleaseScriptRoot '..\..\..'
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
Write-Host "  Architecture:    $script:BuildArch"

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
        Remove-ItemWithoutProgress -LiteralPath $buildDir -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Push-Location $buildDir

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
    Install-ThirdPartySources -Directory $thirdPartyDir -SkipExisting:([bool]$StartAt)

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

    Build-Architecture -Arch $script:BuildArch @buildParams

    Write-Step "Build complete!"
    Write-Host "Packages are in: $buildDir" -ForegroundColor Green

} finally {
    Pop-Location
    # Restore the console mode that was saved at script start.
    if ($null -ne $script:SavedConsoleMode) {
        try { [ConsoleMode]::Set($script:SavedConsoleMode) } catch {}
    }
}
