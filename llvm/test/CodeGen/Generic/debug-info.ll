; RUN: llvm-as < %s | llc

        %lldb.compile_unit = type { i32, i16, i16, i8*, i8*, i8*, {  }* }
@d.compile_unit7 = external global %lldb.compile_unit           ; <%lldb.compile_unit*> [#uses=1]

declare void @llvm.dbg.stoppoint(i32, i32, %lldb.compile_unit*)

define void @rb_raise(i32, ...) {
entry:
        br i1 false, label %strlen.exit, label %no_exit.i

no_exit.i:              ; preds = %entry
        ret void

strlen.exit:            ; preds = %entry
        call void @llvm.dbg.stoppoint( i32 4358, i32 0, %lldb.compile_unit* @d.compile_unit7 )
        unreachable
}

