; RUN: llvm-as < %s |& not grep {Instruction operands must be first-class}

; This previously was for PR826, but structs are now first-class so
; the following is now valid.

        %struct_4 = type { i32 }

define void @test() {
        store %struct_4 zeroinitializer, %struct_4* null
        unreachable
}
