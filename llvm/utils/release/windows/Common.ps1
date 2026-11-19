# Shared release build helpers: step management, common utilities, and version detection.

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
        throw "Required artifact missing: $Description`n  Path: $Path`n  Hint: run the earlier build steps first, or use a different -StartAt value."
    }
}

function Remove-ItemWithoutProgress {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [switch]$Recurse,
        [switch]$Force
    )

    # Recursive Remove-Item reports progress in PowerShell 7. In classic
    # conhost (including cmd.exe), that progress display can leave cursor
    # artifacts behind after deletion completes. This assignment creates a
    # variable local to this function, so the caller's preference is unchanged.
    $ProgressPreference = 'SilentlyContinue'
    Remove-Item -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
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
        Remove-ItemWithoutProgress -LiteralPath $Path -Recurse -Force
    }
}

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
        set(VAR VALUE CACHE <type> "" FORCE) line in a .cmake file in the
        current directory. Returns the file's path, for cmake -C.
    #>
    param(
        [Parameter(Mandatory)]
        [string[]]$Flags,
        [string]$FileName = 'cache_flags.cmake'
    )
    $CacheLines = @()
    foreach ($F in $Flags) {
        if ($F -notmatch '^([A-Za-z_][A-Za-z0-9_]*)(?::([^=]*))?=(.*)$') {
            throw "Not a CMake cache assignment (VAR[:TYPE]=VALUE): '$F'"
        }
        $VarName  = $Matches[1]
        $VarType  = if ($Matches[2]) { $Matches[2] } else { 'STRING' }
        $VarValue = ($Matches[3] -replace '"', '').Replace('\', '/')   # strip quotes, use forward slashes
        $CacheLines += "set($VarName `"$VarValue`" CACHE $VarType `"`" FORCE)"
    }
    $CachePath = (Join-Path $PWD $FileName).Replace('\', '/')
    $CacheLines -join "`n" | Set-Content -Path $CachePath -Encoding UTF8
    return $CachePath
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
    # Keep a real array: a single bare argument would otherwise be splatted
    # character by character.
    $Arguments = @($args | Select-Object -Skip 1)
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

function ConvertTo-UnixPath {
    <#
    .SYNOPSIS
        Convert a Windows path to a Unix-style path. CMake expects forward
        slashes in compiler and install paths.
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
    $Actual = (Get-FileHash -Path $Path -Algorithm $Algorithm).Hash
    if ($Actual -ne $ExpectedHash.ToUpperInvariant()) {
        throw "Checksum mismatch for ${Path}:`n  expected $ExpectedHash`n  actual   $Actual"
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
    $VersionFile = Join-Path $SourceDir 'cmake' 'Modules' 'LLVMVersion.cmake'
    if (-not (Test-Path $VersionFile)) {
        throw "Cannot find LLVMVersion.cmake at: $VersionFile"
    }
    $Content = Get-Content $VersionFile -Raw
    $Major = if ($Content -match 'LLVM_VERSION_MAJOR\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_MAJOR" }
    $Minor = if ($Content -match 'LLVM_VERSION_MINOR\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_MINOR" }
    $Patch = if ($Content -match 'LLVM_VERSION_PATCH\s+(\d+)') { $Matches[1] } else { throw "Cannot parse LLVM_VERSION_PATCH" }
    $Suffix = if ($Content -match 'LLVM_VERSION_SUFFIX\s+(\S+)\)') { $Matches[1] } else { '' }
    # Don't include the "git" development suffix in the release version
    if ($Suffix -eq 'git') { $Suffix = '' }
    return @{
        Major = $Major
        Minor = $Minor
        Patch = $Patch
        Suffix = $Suffix
        Full = "${major}.${minor}.${patch}${suffix}"
    }
}

