; RUN: not llvm-as < %s |& grep {use of undefined type named 'InvalidType'}

define void @test() {
        malloc %InvalidType
        ret void
}

