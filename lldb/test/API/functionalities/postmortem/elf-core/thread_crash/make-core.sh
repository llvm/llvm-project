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

rm -f a.out
make -f main.mk

cat <<EOF
Executable file is in a.out.
Core file will be saved according to pattern $core_pattern.
EOF

# Save stack size and core_dump_filter
stack_size=`ulimit -s`
ulimit -Ss 32 # Decrease stack size to 32k => smaller core files.

core_dump_filter=`cat /proc/self/coredump_filter`
echo 0 > /proc/self/coredump_filter

exec ./a.out

# Reset stack size and core_dump_filter
echo core_dump_filter > /proc/self/coredump_filter
ulimit -s $stack_size
