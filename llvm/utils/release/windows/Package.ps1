# MSI validation and packaging-resume helpers.

function Release-MsiComObject {
    param([object]$ComObject)

    if ($null -ne $ComObject -and [Runtime.InteropServices.Marshal]::IsComObject($ComObject)) {
        try {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($ComObject)
        } catch {
            Write-Warning "Could not release a Windows Installer COM object: $($_.Exception.Message)"
        }
    }
}

function Assert-MsiUpgradeCode {
    param([Parameter(Mandatory)][string]$BuildDirectory)

    # This permanent UpgradeCode must match llvm/CMakeLists.txt.
    $Expected = 'B08613CD-8BD0-4FB6-8937-621936604DE3'
    $MsiFiles = @(Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.msi' -File)
    if ($MsiFiles.Count -ne 1) {
        throw "Expected one MSI in $BuildDirectory, found $($MsiFiles.Count)."
    }

    # Query Windows Installer directly; dark.exe is absent from newer WiX.
    # Close and release every COM object so this process does not keep the MSI
    # open when the workflow moves it after the build script returns.
    $Installer = $null
    $Database = $null
    $View = $null
    $Record = $null
    try {
        $Installer = New-Object -ComObject WindowsInstaller.Installer
        $Database = $Installer.OpenDatabase($MsiFiles[0].FullName, 0)
        $View = $Database.OpenView("SELECT Value FROM Property WHERE Property='UpgradeCode'")
        $View.Execute()
        $Record = $View.Fetch()
        if (-not $Record) {
            throw "Could not read the UpgradeCode from $($MsiFiles[0].FullName)."
        }
        $Actual = $Record.StringData(1).Trim([char[]]'{}')
        if ($Actual -ine $Expected) {
            throw "Unexpected MSI UpgradeCode $Actual in $($MsiFiles[0].FullName); expected $Expected."
        }
        Write-SubStep "Verified MSI UpgradeCode: $Expected"
    } finally {
        Release-MsiComObject $Record
        if ($View) { try { $View.Close() } catch { } }
        Release-MsiComObject $View
        Release-MsiComObject $Database
        Release-MsiComObject $Installer
    }
}

function Test-TarballConfigurationApplied {
    param(
        [Parameter(Mandatory)][string]$BuildDirectory,
        [Parameter(Mandatory)][string]$TarballInstallDirectory
    )

    # The marker covers interrupted tarball steps. Check the cache as well so
    # builds made by earlier script versions can still be resumed.
    $Marker = Join-Path $BuildDirectory '.llvm_release_tarball_configured'
    if (Test-Path -LiteralPath $Marker -PathType Leaf) {
        return $true
    }

    $Cache = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $Cache -PathType Leaf)) {
        return $false
    }

    $InstallPrefix = ConvertTo-UnixPath $TarballInstallDirectory
    $PrefixPattern = '^CMAKE_INSTALL_PREFIX:[^=]+=' + [regex]::Escape($InstallPrefix) + '$'
    $HasTarballInstallPrefix = Select-String -LiteralPath $Cache -Pattern $PrefixPattern -Quiet
    $HasToolchainOnlyDisabled = Select-String -LiteralPath $Cache `
        -Pattern '^LLVM_INSTALL_TOOLCHAIN_ONLY:[^=]+=OFF$' -Quiet
    $HasTestsDisabled = Select-String -LiteralPath $Cache `
        -Pattern '^LLVM_INCLUDE_TESTS:[^=]+=OFF$' -Quiet
    return ($HasTarballInstallPrefix -and $HasToolchainOnlyDisabled -and $HasTestsDisabled)
}
