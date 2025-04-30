#!/bin/bash -e
SCRIPT_PATH=$(cd $(dirname $0) && pwd)

function usage() {
    echo "Run OpenMP nextsilicon unittests."
    echo "Usage: $0 [OPTIONS]"
    echo "  -h, --help                    show this help menu"
    echo "  --debug                       build Debug configuration"
    echo "  --release                     build Release (Default) configuration"
    exit 1
}

CFG=Release

while [[ $# -gt 0 ]] ; do
    key="$1"
    case $key in
        -h|--help)
            usage
        ;;
        --debug)
            CFG=Debug
            shift
        ;;
        --release)
            CFG=Release
            shift
        ;;
        --)
            shift
            break
        ;;
        *)    # unknown option
            echo "Unknown parameter $1" 2>&1
            exit 1
        ;;
    esac
done

pushd ../${CFG}/openmp

ninja check-libomp-nextsilicon

popd 
