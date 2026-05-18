// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%standalone_libs/StandalonePlugin%shlibext --pass-pipeline="builtin.module(standalone-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @standalone_types(%arg0: !standalone.custom<"10">)
  func.func @standalone_types(%arg0: !standalone.custom<"10">) {
    return
  }
}
