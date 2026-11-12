// aarch64-linux-android29-clang -shared -Os -glldb -g3 -Wl,--build-id=sha1 \
//     AndroidModule.c -o AndroidModule.so
// dump_syms AndroidModule.so > AndroidModule.so.sym
// cp AndroidModule.so AndroidModule.unstripped.so
// llvm-strip --strip-unneeded AndroidModule.so

int boom(void) {
  return 47;
}

__attribute__((visibility("hidden"))) int boom_hidden(void) {
  return 48;
}
