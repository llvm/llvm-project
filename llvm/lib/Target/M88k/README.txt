M88k
====

User manuals:
http://www.bitsavers.org/components/motorola/88000/

ELF ABI:
https://archive.org/details/bitsavers_attunixSysa0138776555SystemVRelease488000ABI1990_8011463

Overview calling convention:
https://people.cs.clemson.edu/~mark/subroutines/m88k.html

gcc:
3.3.6 is last version to support m88k
https://gcc.gnu.org/onlinedocs/gcc-3.3.6/gcc/M88K-Options.html#M88K-Options

wget ftp://ftp.nluug.nl/mirror/languages/gcc/releases/gcc-3.3.6/gcc-3.3.6.tar.gz

binutils:
2.16 is last version to support m88k
https://sourceware.org/binutils/docs-2.16/as/M88K_002dDependent.html#M88K_002dDependent

Build:

wget http://ftp.gnu.org/gnu/binutils/binutils-2.16.1a.tar.bz2
tar xjf binutils-2.16.1a.tar.bz2
mkdir build-binutils
cd build-binutils
../binutils-2.16.1/configure --target=m88k-elf --prefix="$HOME/opt/cross" \
    --disable-nls --disable-werror \
    --disable-gdb --disable-libdecnumber --disable-readline --disable-sim

../binutils-2.16.1/configure --target=m88k-unknown-openbsd --prefix="$HOME/opt/cross" \
    --disable-nls --disable-werror \
    --disable-gdb --disable-libdecnumber --disable-readline --disable-sim

#define TARGET_BYTES_BIG_ENDIAN 1
#define TARGET_FORMAT "elf32"
#define TARGET_ARCH  bfd_arch_m88k

git clone git://sourceware.org/git/binutils-gdb.git
cd binutils-gdb
git checkout binutils-2_16_1

../binutils-gdb/configure --enable-targets=m88k-openbsd-elf32 --prefix="$HOME/cross" --disable-nls

../binutils-gdb/configure --targets=all

../binutils-gdb/configure --target=m88k-openbsd --prefix="$HOME/opt/cross" \
    --disable-nls --disable-werror \
    --disable-gdb --disable-libdecnumber --disable-readline --disable-sim

../binutils-gdb/configure --target=m88k-openbsd --prefix="$HOME/opt/cross" \
    --disable-nls --disable-werror \
    --disable-gdb --disable-libdecnumber --disable-readline --disable-sim


Testing with GXemul: http://gavare.se/gxemul/

On FreeBSD:

To enable tap device, add the following:

In /etc/sysctl.conf:
net.link.tap.user_open=1

In /etc/rc.conf:
cloned_interfaces="bridge0 tap0"
ifconfig_bridge0="addm em0 addm tap0 up"

In /boot/loader.conf:
if_bridge_load="YES"
if_tap_load="YES"

Using config file lunam88k.cfg for GXemul:
net(
        tapdev("/dev/tap0")
)
machine(
        type("luna88k")
        subtype("luna-88k")
        disk("liveimage-luna88k-raw-20210918.img")
        load("boot")
)
