; RUN: llvm-as < %s | llc -march=x86-64 | grep rep.movsq | count 2
; RUN: llvm-as < %s | llc -march=x86 | grep rep.movsl	 | count 2

%struct.s = type { i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16, i16, i16, i16, i16, i16, i16, i16,
                   i16 }


define void @g(i16 signext  %a1, i16 signext  %a2, i16 signext  %a3,
	 i16 signext  %a4, i16 signext  %a5, i16 signext  %a6) nounwind {
entry:
        %a = alloca %struct.s, align 16
        %tmp = getelementptr %struct.s* %a, i32 0, i32 0
        store i16 %a1, i16* %tmp, align 16
        %tmp2 = getelementptr %struct.s* %a, i32 0, i32 1
        store i16 %a2, i16* %tmp2, align 16
        %tmp4 = getelementptr %struct.s* %a, i32 0, i32 2
        store i16 %a3, i16* %tmp4, align 16
        %tmp6 = getelementptr %struct.s* %a, i32 0, i32 3
        store i16 %a4, i16* %tmp6, align 16
        %tmp8 = getelementptr %struct.s* %a, i32 0, i32 4
        store i16 %a5, i16* %tmp8, align 16
        %tmp10 = getelementptr %struct.s* %a, i32 0, i32 5
        store i16 %a6, i16* %tmp10, align 16
        call void @f( %struct.s* %a byval )
        call void @f( %struct.s* %a byval )
        ret void
}

declare void @f(%struct.s* byval)
