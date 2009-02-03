// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct face_cachel {
  unsigned int reverse :1;
  unsigned char font_specified[1];
};

void
ensure_face_cachel_contains_charset (struct face_cachel *cachel) {
  cachel->font_specified[0] = 0;
}

