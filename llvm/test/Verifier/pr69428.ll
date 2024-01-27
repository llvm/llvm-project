; RUN: llvm-as -disable-output %s

%struct._List_node_emplace_op2 = type { i8 }

$"??1?$_List_node_emplace_op2@H@@QEAA@XZ" = comdat any

@"?_List@@3HA" = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress noreturn
define dso_local void @"?ExecutionEngineaddExecutableDependency@@YAXXZ"() local_unnamed_addr #0 personality ptr @__CxxFrameHandler3 {
entry:
  %agg.tmp.ensured.i = alloca %struct._List_node_emplace_op2, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %agg.tmp.ensured.i)
  %0 = load i32, ptr @"?_List@@3HA", align 4
  %call.i = call noundef ptr @"??0?$_List_node_emplace_op2@H@@QEAA@H@Z"(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp.ensured.i, i32 noundef %0)
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont.i unwind label %ehcleanup.i

invoke.cont.i:                                    ; preds = %entry
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont2.i unwind label %ehcleanup.i

invoke.cont2.i:                                   ; preds = %invoke.cont.i
  call void @"??1?$_List_node_emplace_op2@H@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp.ensured.i) #6
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
  call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp.ensured.i) #6 [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller

"??1?$_List_node_emplace_op2@H@@QEAA@XZ.exit.i":  ; preds = %invoke.cont.i.i
  call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp.ensured.i) #6 [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

declare dso_local noundef ptr @"??0?$_List_node_emplace_op2@H@@QEAA@H@Z"(ptr noundef nonnull returned align 1 dereferenceable(1), i32 noundef) unnamed_addr #1

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nofree nosync nounwind memory(none)
declare dso_local void @llvm.seh.scope.begin() #2

; Function Attrs: nofree nosync nounwind memory(none)
declare dso_local void @llvm.seh.scope.end() #2

; Function Attrs: mustprogress nounwind
define linkonce_odr dso_local void @"??1?$_List_node_emplace_op2@H@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @llvm.seh.scope.begin()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @llvm.seh.scope.end()
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont
  tail call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %this) #6
  ret void

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %0 = cleanuppad within none []
  call void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %this) #6 [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; Function Attrs: nounwind
declare dso_local void @"??1_Alloc_construct_ptr@@QEAA@XZ"(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5
