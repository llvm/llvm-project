; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; END.

	%arraytype.1.Char = type { i32, [0 x i8] }
	%arraytype.4.Signed = type { i32, [0 x i32] }
	%functiontype.23 = type %structtype.Task* (%structtype.Task*, %structtype.Packet*, %structtype.FailedRun*)
	%functiontype.27 = type %structtype.object* ()
	%functiontype.28 = type i1 (%structtype.object*, %structtype.object_vtable*)
	%functiontype.39 = type i32 (%structtype.listiter*)
	%opaquetype.RuntimeTypeInfo = type i8* (i8*)
	%structtype.AssertionError_vtable = type { %structtype.FailedRun_vtable }
	%structtype.DeviceTask = type { %structtype.Task }
	%structtype.FailedRun = type { %structtype.object }
	%structtype.FailedRun_vtable = type { %structtype.object_vtable }
	%structtype.Packet = type { %structtype.object, %structtype.list.1*, i32, i32, i32, %structtype.Packet* }
	%structtype.Task = type { %structtype.TaskState, %structtype.FailedRun*, i32, %structtype.Packet*, %structtype.Task*, i32 }
	%structtype.TaskState = type { %structtype.object, i1, i1, i1 }
	%structtype.list.1 = type { %arraytype.4.Signed* }
	%structtype.listiter = type { %structtype.list.1*, i32 }
	%structtype.object = type { %structtype.object_vtable* }
	%structtype.object_vtable = type { %structtype.object_vtable*, %opaquetype.RuntimeTypeInfo*, %arraytype.1.Char*, %functiontype.27* }
@structinstance.59 = external global %structtype.AssertionError_vtable		; <%structtype.AssertionError_vtable*> [#uses=0]

declare fastcc i1 @ll_isinstance__objectPtr_object_vtablePtr()

declare fastcc void @ll_listnext__listiterPtr()

define fastcc void @WorkTask.fn() {
block0:
	br label %block1
block1:		; preds = %block0
	%v2542 = call fastcc i1 @ll_isinstance__objectPtr_object_vtablePtr( )		; <i1> [#uses=1]
	br i1 %v2542, label %block4, label %block2
block2:		; preds = %block1
	br label %block3
block3:		; preds = %block2
	unwind
block4:		; preds = %block1
	br label %block5
block5:		; preds = %block4
	%v2565 = icmp eq %structtype.Packet* null, null		; <i1> [#uses=1]
	br i1 %v2565, label %block15, label %block6
block6:		; preds = %block5
	%self_2575 = phi %structtype.DeviceTask* [ null, %block5 ]		; <%structtype.DeviceTask*> [#uses=1]
	br i1 false, label %block14, label %block7
block7:		; preds = %block14, %block6
	%self_2635 = phi %structtype.DeviceTask* [ %self_2575, %block6 ], [ null, %block14 ]		; <%structtype.DeviceTask*> [#uses=1]
	%tmp.124 = getelementptr %structtype.Packet* null, i32 0, i32 2		; <i32*> [#uses=0]
	br label %block8
block8:		; preds = %block10, %block7
	%self_2672 = phi %structtype.DeviceTask* [ %self_2635, %block7 ], [ null, %block10 ]		; <%structtype.DeviceTask*> [#uses=0]
	invoke fastcc void @ll_listnext__listiterPtr( )
			to label %block9 unwind label %block8_exception_handling
block8_exception_handling:		; preds = %block8
	br i1 false, label %block8_exception_found_branchto_block12, label %block8_not_exception_structinstance.10
block8_not_exception_structinstance.10:		; preds = %block8_exception_handling
	unwind
block8_exception_found_branchto_block12:		; preds = %block8_exception_handling
	br label %block12
block9:		; preds = %block8
	br i1 false, label %block11, label %block10
block10:		; preds = %block11, %block9
	br label %block8
block11:		; preds = %block9
	br label %block10
block12:		; preds = %block8_exception_found_branchto_block12
	br label %block13
block13:		; preds = %block15, %block12
	ret void
block14:		; preds = %block6
	br label %block7
block15:		; preds = %block5
	%v2586 = phi %structtype.DeviceTask* [ null, %block5 ]		; <%structtype.DeviceTask*> [#uses=0]
	br label %block13
}
