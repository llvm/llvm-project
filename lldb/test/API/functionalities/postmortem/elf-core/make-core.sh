#! /bin/sh

linux_check_core_pattern()
{
    if grep -q '^|' </proc/sys/kernel/core_pattern; then
        cat <<EOF
Your system uses a crash report tool ($(cat /proc/sys/kernel/core_pattern)). Core files
will not be generated.  Please reset /proc/sys/kernel/core_pattern (requires root
privileges) to enable core generation.
EOF
        exit 1
    fi
}

OS=$(uname -s)
case "$OS" in
FreeBSD)
    core_pattern=$(sysctl -n kern.corefile)
    ;;
Linux)
    core_pattern=$(cat /proc/sys/kernel/core_pattern)
    ;;
*)
    echo "OS $OS not supported" >&2
    exit 1
    ;;
esac

set -e -x

file=$1
if [ -z "$file" ]; then
    cat <<EOF
Please supply the main source file as the first argument.
EOF
    exit 1
fi

if [ "$OS" = Linux ]; then
    linux_check_core_pattern
fi

ulimit -c 1000
real_limit=$(ulimit -c)
if [ $real_limit -lt 100 ]; then
    cat <<EOF
Unable to increase the core file limit. Core file may be truncated!
To fix this, increase HARD core file limit (ulimit -H -c 1000). This may require root
privileges.
EOF
fi

${CC:-cc} -nostdlib -static -g $CFLAGS "$file" -o a.out

cat <<EOF
Executable file is in a.out.
Core file will be saved according to pattern $core_pattern.
EOF

ulimit -s 8 # Decrease stack size to 8k => smaller core files.
exec ./a.out
