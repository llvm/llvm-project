

testparametric.func @callee(%arg0: !testparametric.param<"A"> ) attributes { metaParams = ["A", "B"]} {
    %value = testparametric.add %arg0, %arg0 : (!testparametric.param<"A">, !testparametric.param<"A">) -> !testparametric.param<"A">
    testparametric.print_attr #testparametric.param<"B">
    return
}

func.func @caller() {
    %cst0 = arith.constant 0 : i32
    %cst1 = arith.constant 1. : f32
    %cst2 = arith.constant 2. : f64
    testparametric.call @callee(%cst0) meta = {"A" = i32, "B" = 32 : i64 } : (i32) -> ()
    testparametric.call @callee(%cst0) meta = {"A" = i32, "B" = 32 : i64 } : (i32) -> ()
    testparametric.call @callee(%cst1) meta = {"A" = f32, "B" = 64 : i64 } : (f32) -> ()
    testparametric.call @callee(%cst2) meta = {"A" = f64, "B" = 128 : i64 } : (f64) -> ()
    return
}
