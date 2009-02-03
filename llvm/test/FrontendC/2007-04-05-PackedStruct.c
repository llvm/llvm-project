// RUN: %llvmgcc %s -S -o -

#pragma pack(push, 2)

enum {
  tA = 0,
  tB = 1
};

struct MyStruct {
  unsigned long A;
  char C;
  void * B;
};

void bar(){
struct MyStruct MS = { tB, 0 };
}
