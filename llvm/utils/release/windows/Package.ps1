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
    $expected = 'B08613CD-8BD0-4FB6-8937-621936604DE3'
    $msiFiles = @(Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.msi' -File)
    if ($msiFiles.Count -ne 1) {
        throw "Expected one MSI in $BuildDirectory, found $($msiFiles.Count)."
    }

    # Query Windows Installer directly; dark.exe is absent from newer WiX.
    # Close and release every COM object so this process does not keep the MSI
    # open when the workflow moves it after the build script returns.
    $installer = $null
    $database = $null
    $view = $null
    $record = $null
    try {
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
    } finally {
        Release-MsiComObject $record
        if ($view) { try { $view.Close() } catch { } }
        Release-MsiComObject $view
        Release-MsiComObject $database
        Release-MsiComObject $installer
    }
}

function Test-TarballConfigurationApplied {
    param(
        [Parameter(Mandatory)][string]$BuildDirectory,
        [Parameter(Mandatory)][string]$TarballInstallDirectory
    )

    # The marker covers interrupted tarball steps. Check the cache as well so
    # builds made by earlier script versions can still be resumed.
    $marker = Join-Path $BuildDirectory '.llvm_release_tarball_configured'
    if (Test-Path -LiteralPath $marker -PathType Leaf) {
        return $true
    }

    $cache = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
        return $false
    }

    $installPrefix = ConvertTo-ForwardSlashPath $TarballInstallDirectory
    $prefixPattern = '^CMAKE_INSTALL_PREFIX:[^=]+=' + [regex]::Escape($installPrefix) + '$'
    $hasTarballInstallPrefix = Select-String -LiteralPath $cache -Pattern $prefixPattern -Quiet
    $hasToolchainOnlyDisabled = Select-String -LiteralPath $cache `
        -Pattern '^LLVM_INSTALL_TOOLCHAIN_ONLY:[^=]+=OFF$' -Quiet
    $hasTestsDisabled = Select-String -LiteralPath $cache `
        -Pattern '^LLVM_INCLUDE_TESTS:[^=]+=OFF$' -Quiet
    return ($hasTarballInstallPrefix -and $hasToolchainOnlyDisabled -and $hasTestsDisabled)
}
