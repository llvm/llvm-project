; RUN: llvm-as -disable-output %s

%struct._List_node_emplace_op2 = type { i8 }

@"?_List@@3HA" = global i32 0, align 4

define void @"?ExecutionEngineaddExecutableDependency@@YAXXZ"() personality ptr @__CxxFrameHandler3 {
entry:
  %agg.tmp.ensured.i = alloca %struct._List_node_emplace_op2, align 1
  %0 = load i32, ptr @"?_List@@3HA", align 4
  %call.i = call noundef ptr @"??0?$_List_node_emplace_op2@H@@QEAA@H@Z"(ptr %agg.tmp.ensured.i, i32 %0)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont.i unwind label %ehcleanup.i

invoke.cont.i:                                    ; preds = %entry
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont2.i unwind label %ehcleanup.i

invoke.cont2.i:                                   ; preds = %invoke.cont.i
  call void @"??1?$_List_node_emplace_op2@H@@QEAA@XZ"(ptr %agg.tmp.ensured.i) #6
  unreachable

ehcleanup.i:                                      ; preds = %invoke.cont.i, %entry
  %1 = cleanuppad within none []
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont.i.i unwind label %ehcleanup.i.i

invoke.cont.i.i:                                  ; preds = %ehcleanup.i
  invoke void @llvm.seh.scope.end()
          to label %"??1?$_List_node_emplace_op2@H@@QEAA@XZ.exit.i" unwind label %ehcleanup.i.i

ehcleanup.i.i:                                    ; preds = %invoke.cont.i.i, %ehcleanup.i
  %2 = cleanuppad within %1 []
  call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr %agg.tmp.ensured.i) #6 [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller

"??1?$_List_node_emplace_op2@H@@QEAA@XZ.exit.i":  ; preds = %invoke.cont.i.i
  call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr %agg.tmp.ensured.i) #6 [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

declare i32 @__CxxFrameHandler3(...)
declare void @llvm.seh.scope.begin()
declare void @llvm.seh.scope.end()

declare void @"??1?$_List_node_emplace_op2@H@@QEAA@XZ"(ptr)
declare void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr)
declare ptr @"??0?$_List_node_emplace_op2@H@@QEAA@H@Z"(ptr, i32)
