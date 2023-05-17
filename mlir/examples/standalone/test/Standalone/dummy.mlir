// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %res = standalone.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @standalone_types(%arg0: !standalone.custom<"10">)
    func.func @standalone_types(%arg0: !standalone.custom<"10">) {
        return
    }
}
