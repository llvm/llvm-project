; RUN: BrainF %s -o - | llvm-dis | FileCheck %s
++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.

# CHECK-LABEL: define void @brainf() {
# CHECK-NEXT:      brainf:
# CHECK-NEXT:        %malloccall = tail call i8* @malloc(i32 mul (i32 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i32), i32 65536))
# CHECK-NEXT:        call void @llvm.memset.p0i8.i32(i8* %malloccall, i8 0, i32 65536, i1 false)
# CHECK-NEXT:        %head = getelementptr i8, i8* %malloccall, i32 32768
# CHECK-NEXT:        %tape = load i8, i8* %head, align 1
# CHECK:             store i8 %tape2, i8* %head, align 1
# CHECK:             br label %brainf3
# CHECK:           brainf1:                                          ; preds = %brainf55
# CHECK-NEXT:        tail call void @free(i8* %malloccall)
# CHECK-NEXT:        ret void
# CHECK:       }
# CHECK-LABEL: define i32 @main(i32 %argc, i8** %argv) {
# CHECK-NEXT:      main.0:
# CHECK-NEXT:        call void @brainf()
# CHECK-NEXT:        ret i32 0
# CHECK:       }
