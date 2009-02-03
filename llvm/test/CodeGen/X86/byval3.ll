; RUN: llvm-as < %s | llc -march=x86-64 | grep rep.movsq | count 2
; RUN: llvm-as < %s | llc -march=x86 | grep rep.movsl | count 2

%struct.s = type { i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32 }

define void @g(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6) nounwind {
entry:
        %d = alloca %struct.s, align 16
        %tmp = getelementptr %struct.s* %d, i32 0, i32 0
        store i32 %a1, i32* %tmp, align 16
        %tmp2 = getelementptr %struct.s* %d, i32 0, i32 1
        store i32 %a2, i32* %tmp2, align 16
        %tmp4 = getelementptr %struct.s* %d, i32 0, i32 2
        store i32 %a3, i32* %tmp4, align 16
        %tmp6 = getelementptr %struct.s* %d, i32 0, i32 3
        store i32 %a4, i32* %tmp6, align 16
        %tmp8 = getelementptr %struct.s* %d, i32 0, i32 4
        store i32 %a5, i32* %tmp8, align 16
        %tmp10 = getelementptr %struct.s* %d, i32 0, i32 5
        store i32 %a6, i32* %tmp10, align 16
        call void @f( %struct.s* %d byval)
        call void @f( %struct.s* %d byval)
        ret void
}

declare void @f(%struct.s* byval)
