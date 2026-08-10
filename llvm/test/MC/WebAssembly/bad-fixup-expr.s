// RUN: not llvm-mc -filetype=obj -triple wasm32 %s -o /dev/null 2>&1 | FileCheck %s

  .section    .data.foo,"",@
foo:
  .int8       1
  .size       foo, 1
foo_other:
  .int8       1
  .size       foo_other, 1

  .section    .data.bar,"",@
bar:
  .int8       1
  .size       bar, 1

  .section    .data.fizz,"",@
fizz:
  .int8       1
  .size       fizz, 1

  .section    .data.segment1,"",@
segment1:
// CHECK: 'bar' can not be placed in a different section
  .int32 fizz-bar
// CHECK: 'undef_baz' can not be undefined in a subtraction expression
  .int32 fizz-undef_baz
// CHECK: 'fizz' can not be placed in a different section
  .int32 undef_baz-fizz
  .size       segment1, 12


  .text
  .section    .text.main,"",@
main:
  .functype   main () -> (i32, i32, i32, i32, i32)
// Expressions involving symbols within the same sections can be evaluated
// prior to writing the object file.
// CHECK-NOT: foo
  i32.const foo-foo_other+2
  i32.const foo_other-foo-10

// CHECK: 'bar' unsupported subtraction expression used in relocation in code section.
  i32.const foo-bar
// CHECK: 'undef_baz' unsupported subtraction expression used in relocation in code section.
  i32.const foo-undef_baz
// CHECK: 'foo' unsupported subtraction expression used in relocation in code section.
  i32.const undef_baz-foo
  end_function
