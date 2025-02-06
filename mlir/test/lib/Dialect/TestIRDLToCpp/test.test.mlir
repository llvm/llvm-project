func.func @test() {
    %1 = arith.constant 5 : i32
    "test_irdl_to_cpp.bar"() : () -> i32
    return
}