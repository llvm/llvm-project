#! /bin/sh

linux_check_ptrace_scope()
{
    if grep -q '1' </proc/sys/kernel/yama/ptrace_scope; then
        cat <<EOF
Your system prevents the use of PTRACE to attach to non-child processes. The core file
cannot be generated.  Please reset /proc/sys/kernel/yama/ptrace_scope to 0 (requires root
privileges) to enable core generation via gcore.
EOF
        exit 1
    fi
}

set -e -x

OS=$(uname -s)
if [ "$OS" = Linux ]; then
    linux_check_ptrace_scope
fi

rm -f a.out
make -f main.mk

cat <<EOF
Executable file is in a.out.
Core file will be saved as core.<pid>.
EOF

stack_size=`ulimit -s`

# Decrease stack size to 16k => smaller core files.
# gcore won't run with the smaller stack
ulimit -Ss 16

core_dump_filter=`cat /proc/self/coredump_filter`
echo 0 > /proc/self/coredump_filter

./a.out &

pid=$!

echo $core_dump_filter > /proc/self/coredump_filter

# Reset stack size as so there's enough space to run gcore.
ulimit -s $stack_size

echo "Sleeping for 5 seconds to wait for $pid"

sleep 5
echo "Taking core from process $pid"

gcore -o core $pid

echo "Killing process $pid"
kill -9 $pid
