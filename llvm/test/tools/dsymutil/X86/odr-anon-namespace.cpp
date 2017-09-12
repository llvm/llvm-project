/* Compile with:
   for FILE in `seq 2`; do
     clang -g -c  odr-anon-namespace.cpp -DFILE$FILE -o odr-anon-namespace/$FILE.o
   done
 */

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/odr-anon-namespace -y %p/dummy-debug-map.map -o - | llvm-dwarfdump -debug-info - | FileCheck %s

#ifdef FILE1
// Currently llvm-dsymutil will unique the contents of anonymous
// namespaces if they are from the same file/line. Force this
// namespace to appear different eventhough it's the same (this
// uniquing is actually a bug kept for backward compatibility, see the
// comments in DeclContextTree::getChildDeclContext()).
#line 42
#endif
namespace {
class C {};
}

void foo() {
  C c;
}

// Keep the ifdef guards for FILE1 and FILE2 even if all the code is
// above to clearly show what the CHECK lines are testing.
#ifdef FILE1

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-anon-namespace.cpp"

// CHECK: DW_TAG_variable
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name {{.*}}"c"
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_type {{.*}}0x00000000[[C_FILE1:[0-9a-f]*]]

// CHECK: DW_TAG_namespace
// CHECK-NOT: {{DW_AT_name|NULL|DW_TAG}}
// CHECK: 0x[[C_FILE1]]:{{.*}}DW_TAG_class_type
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name{{.*}}"C"

#elif defined(FILE2)

// CHECK: TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: AT_name{{.*}}"odr-anon-namespace.cpp"

// CHECK: DW_TAG_variable
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name {{.*}}"c"
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_type {{.*}}0x00000000[[C_FILE2:[0-9a-f]*]]

// CHECK: DW_TAG_namespace
// CHECK-NOT: {{DW_AT_name|NULL|DW_TAG}}
// CHECK: 0x[[C_FILE2]]:{{.*}}DW_TAG_class_type
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name{{.*}}"C"

#else
#error "You must define which file you generate"
#endif
