# Third-party dependency build helpers.

$script:ThirdPartyLibraries = @(
    @{
        Name = 'libxml2'
        SourceDirectory = 'libxml2-v2.15.3'
        Archive = 'libxml2-v2.15.3.tar.gz'
        DownloadArguments = @('--remote-name', 'https://gitlab.gnome.org/GNOME/libxml2/-/archive/v2.15.3/libxml2-v2.15.3.tar.gz')
        Sha256 = '0DA50C1415F4EC0364569D2119B1436BA837B31DF44AF28569D234272C23CF1F'
        ExtractArguments = @('zxf', 'libxml2-v2.15.3.tar.gz', '--exclude', 'test/*')
        BuildFunction = 'Build-LibXml2'
    }
    @{
        Name = 'zlib'
        SourceDirectory = 'zlib-1.3.2'
        Archive = 'zlib-1.3.2.tar.gz'
        DownloadArguments = @('-LO', 'https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz')
        Sha256 = 'BB329A0A2CD0274D05519D61C667C062E06990D72E125EE2DFA8DE64F0119D16'
        ExtractArguments = @('zxf', 'zlib-1.3.2.tar.gz')
        BuildFunction = 'Build-Zlib'
    }
    @{
        Name = 'zstd'
        SourceDirectory = 'zstd-1.5.7'
        Archive = 'zstd-1.5.7.tar.gz'
        DownloadArguments = @('-LO', 'https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz')
        Sha256 = 'EB33E51F49A15E023950CD7825CA74A4A2B43DB8354825AC24FC1B7EE09E6FA3'
        ExtractArguments = @('zxf', 'zstd-1.5.7.tar.gz', '--exclude', 'tests/*')
        BuildFunction = 'Build-Zstd'
    }
)

function Install-ThirdPartySources {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [switch]$SkipExisting
    )

    Push-Location $Directory
    try {
        foreach ($library in $script:ThirdPartyLibraries) {
            if ($SkipExisting -and (Test-Path -LiteralPath $library.SourceDirectory)) {
                Write-Step "Skipping $($library.Name) download (already exists)"
                continue
            }

            Write-Step "Downloading $($library.Name)"
            $downloadArguments = $library.DownloadArguments
            Invoke-NativeCommand curl.exe @downloadArguments
            Test-FileChecksum -Path $library.Archive -Algorithm SHA256 `
                -ExpectedHash $library.Sha256
            $extractArguments = $library.ExtractArguments
            Invoke-NativeCommand tar @extractArguments
        }
    } finally {
        Pop-Location
    }
}

function Build-ThirdPartyLibraries {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string[]]$ToolchainFlags
    )

    foreach ($library in $script:ThirdPartyLibraries) {
        $sourceDirectory = Join-Path $Directory $library.SourceDirectory
        $buildFunction = $library.BuildFunction
        & $buildFunction -SourceDir $sourceDirectory -ToolchainFlags $ToolchainFlags
    }
}

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
            # Suppress warnings in the downloaded dependencies with -w; clang,
            # clang-cl and cl all accept it.
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
        $script:LibXmlInstallDir = ConvertTo-ForwardSlashPath (Join-Path $PWD 'install')
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
        $script:ZlibInstallDir = ConvertTo-ForwardSlashPath (Join-Path $PWD 'install')
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
            # This is required so that the /MT flag below takes effect.
            "CMAKE_POLICY_DEFAULT_CMP0091=NEW"
            "CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        )
        $cache = Write-CMakeCacheFile -Flags $zstdFlags -FileName 'zstd_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$SourceDir/build/cmake"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install
        $script:ZstdInstallDir = ConvertTo-ForwardSlashPath (Join-Path $PWD 'install')
    } finally {
        Pop-Location
    }
}

