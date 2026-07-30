#include "B.h" // -*- ObjC -*-

typedef int Typedef;

struct TopLevelStruct {
  int a;
};

typedef struct Struct_s {
  int a;
} Struct;

struct Nested {
  StructB fromb;
};

typedef enum Enum_e { a = 0 } Enum;

@interface SomeClass {
}
@property (readonly) int number;
@end

template <typename T> struct Template { T field; };
extern template struct Template<int>;

namespace Namespace {
template <typename T> struct InNamespace { T field; };
extern template struct InNamespace<int>;
}
