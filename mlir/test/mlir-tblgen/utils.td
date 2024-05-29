// RUN: mlir-tblgen -I %S/../../include %s | FileCheck %s

include "mlir/IR/Utils.td"

// CHECK-DAG: string value = "CamelCaseTest"
class already_camel_case {
  string value = snakeCaseToCamelCase<"CamelCaseTest">.ret;
}

// CHECK-DAG: string value = "Foo"
class single_word {
  string value = snakeCaseToCamelCase<"foo">.ret;
}

// CHECK-DAG: string value = "ThisIsATest"
class snake_case {
  string value = snakeCaseToCamelCase<"this_is_a_test">.ret;
}

// CHECK-DAG: string value = "ThisIsATestAgain"
class extra_underscores {
  string value = snakeCaseToCamelCase<"__this__is_a_test__again__">.ret;
}
