int _start() {
    return 0;
}

// Compile with "clang --target=x86_64-pc-linux -c -g symbolize-64bit-addr.c".
// Link with "ld.lld -Ttext=0xffffffff00000000 symbolize-64bit-addr.o -o symbolize-64bit-addr.elf.x86_64".
