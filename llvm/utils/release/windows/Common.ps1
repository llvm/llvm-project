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

function Remove-ItemWithoutProgress {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [switch]$Recurse,
        [switch]$Force
    )

    # Recursive Remove-Item reports progress in PowerShell 7. In classic
    # conhost (including cmd.exe), that progress display can leave cursor
    # artifacts behind after deletion completes.
    $savedProgressPreference = $ProgressPreference
    try {
        $ProgressPreference = 'SilentlyContinue'
        Remove-Item -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
    } finally {
        $ProgressPreference = $savedProgressPreference
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

function ConvertTo-ForwardSlashPath {
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

