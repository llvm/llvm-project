; RUN: llvm-as < %s | llc -march=c

declare void @llvm.va_end(i8*)

define void @test() {
        %va.upgrd.1 = bitcast i8* null to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.1 )
        ret void
}

