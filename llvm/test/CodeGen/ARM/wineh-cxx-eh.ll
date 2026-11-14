; RUN: llc -mtriple=thumbv7-pc-windows-msvc < %s | FileCheck %s

%rtti.TypeDescriptor7 = type { ptr, ptr, [8 x i8] }
%struct.E = type { i32 }
$"??_R0?AUE@@@8" = comdat any
@"??_7type_info@@6B@" = external constant ptr
@"??_R0?AUE@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUE@@\00" }, comdat
@vec = global <4 x i32> zeroinitializer, align 16
@idx = global i32 0
@out = global i32 0

declare void @g()
declare i32 @use(i32)
declare void @dtor(ptr)
declare i32 @__CxxFrameHandler3(...)

; catch (...) without a catch object. The code after the catch stays in the
; parent function.
define i32 @catch_all() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: catch_all:
; CHECK:       push {r6, lr}
; CHECK:       sub sp, #[[#CA_SUB:]]
; CHECK:       .seh_endprologue
; CHECK-NEXT:  mov r6, sp
; CHECK-NEXT:  mvn [[CA_T:r[0-9]+]], #1
; CHECK-NEXT:  str [[CA_T]], [r6]
; CHECK-NEXT:  str [[CA_T]], [r6, #4]
; CHECK:       [[CA_CONT:\$MBB[0-9_]+]]: @ Block address taken
; CHECK:       .seh_endproc
; CHECK-LABEL: "?catch$3@?0?catch_all@4HA":
; CHECK:       .seh_handler __CxxFrameHandler3, %unwind, %except
; CHECK:       movw r0, :lower16:[[CA_CONT]]
; CHECK-NEXT:  movt r0, :upper16:[[CA_CONT]]
; CHECK-NEXT:  .seh_startepilogue
; CHECK:       bx lr
; CHECK:       $cppxdata$catch_all:
; CHECK:       .long [[#%d,0 - CA_SUB - 8]] @ UnwindHelp
; CHECK-NEXT:  .long 0 @ ESTypeList
; CHECK-NEXT:  .long 1 @ EHFlags
; CHECK:       .long 64 @ Adjectives
; CHECK-NEXT:  .long 0 @ Type
; CHECK-NEXT:  .long 0 @ CatchObjOffset
; CHECK-NEXT:  .long "?catch$3@?0?catch_all@4HA"@imgrel @ Handler
entry:
  invoke void @g() to label %done unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %done
done:
  %r = phi i32 [ 1, %catch ], [ 0, %entry ]
  ret i32 %r
}

; Two catch handlers. 32-bit HandlerType entries are 16 bytes: there is no
; ParentFrameOffset, which only the 64-bit layout has.
define i32 @two_catches() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: two_catches:
; CHECK:       $handlerMap$0$two_catches:
; CHECK-NEXT:  .long 8 @ Adjectives
; CHECK-NEXT:  .long "??_R0?AUE@@@8"@imgrel @ Type
; CHECK-NEXT:  .long {{-[0-9]+}} @ CatchObjOffset
; CHECK-NEXT:  .long "?catch${{[0-9]+}}@?0?two_catches@4HA"@imgrel @ Handler
; CHECK-NEXT:  .long 64 @ Adjectives
; CHECK-NEXT:  .long 0 @ Type
; CHECK-NEXT:  .long 0 @ CatchObjOffset
; CHECK-NEXT:  .long "?catch${{[0-9]+}}@?0?two_catches@4HA"@imgrel @ Handler
; CHECK-NOT:   ParentFrameOffset
entry:
  %e = alloca ptr, align 4
  invoke void @g() to label %done unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch.e, label %catch.all] unwind to caller
catch.e:
  %cp1 = catchpad within %cs [ptr @"??_R0?AUE@@@8", i32 8, ptr %e]
  catchret from %cp1 to label %done
catch.all:
  %cp2 = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp2 to label %done
done:
  %r = phi i32 [ 0, %entry ], [ 1, %catch.e ], [ 2, %catch.all ]
  ret i32 %r
}

; The code after the catch is reached only through the catchret.
define void @only_catch() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: only_catch:
; CHECK:       [[OC_CONT:\$MBB[0-9_]+]]: @ Block address taken
; CHECK:       bl g
; CHECK:       .seh_endproc
; CHECK-LABEL: "?catch$3@?0?only_catch@4HA":
; CHECK-NOT:   bl g
; CHECK:       movw r0, :lower16:[[OC_CONT]]
; CHECK-NEXT:  movt r0, :upper16:[[OC_CONT]]
; CHECK-NOT:   bl g
; CHECK:       .seh_endproc
entry:
  invoke void @g() to label %unreachable unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %done
done:
  call void @g()
  ret void
unreachable:
  unreachable
}

; catch (E e) by value: the funclet reads the catch object where the runtime
; copied it.
define i32 @catch_obj() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: catch_obj:
; CHECK:       push.w {r4, r6, r11, lr}
; CHECK-NEXT:  .seh_save_regs_w {r4, r6, r11, lr}
; CHECK-NEXT:  sub sp, #[[#CB_SUB:]]
; CHECK:       .seh_endprologue
; CHECK-NEXT:  mov r6, sp
; CHECK-NEXT:  mvn [[CB_T:r[0-9]+]], #1
; CHECK-NEXT:  str [[CB_T]], [r6]
; CHECK-NEXT:  str [[CB_T]], [r6, #4]
; CHECK:       [[CB_CONT:\$MBB[0-9_]+]]: @ Block address taken
; CHECK-LABEL: "?catch$3@?0?catch_obj@4HA":
; CHECK:       ldr r0, [r6, #[[#CB_OBJ:]]]
; CHECK:       bl use
; CHECK:       bl dtor
; CHECK:       movw r0, :lower16:[[CB_CONT]]
; CHECK-NEXT:  movt r0, :upper16:[[CB_CONT]]
; CHECK:       $cppxdata$catch_obj:
; CHECK:       .long [[#%d,0 - CB_SUB - 16]] @ UnwindHelp
; CHECK:       .long "??_R0?AUE@@@8"@imgrel @ Type
; CHECK-NEXT:  .long [[#%d,CB_OBJ - CB_SUB - 16]] @ CatchObjOffset
entry:
  %e = alloca %struct.E, align 4
  invoke void @g() to label %done unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr @"??_R0?AUE@@@8", i32 0, ptr %e]
  %v = load i32, ptr %e, align 4
  %u = invoke i32 @use(i32 %v) [ "funclet"(token %cp) ]
          to label %catch.cont unwind label %catch.cleanup
catch.cont:
  call void @dtor(ptr %e) [ "funclet"(token %cp) ]
  catchret from %cp to label %done
catch.cleanup:
  %cl = cleanuppad within %cp []
  call void @dtor(ptr %e) [ "funclet"(token %cl) ]
  cleanupret from %cl unwind to caller
done:
  %r = phi i32 [ %u, %catch.cont ], [ 0, %entry ]
  ret i32 %r
}

; A parent local read and written in the catch, with a frame pointer: the
; funclet sets up its own r11, so the local must be reached through r6.
define i32 @catch_parent_local(i32 %x) "frame-pointer"="all" personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: catch_parent_local:
; CHECK:       str r0, [r6, #[[#PL:]]]
; CHECK-LABEL: "?catch$2@?0?catch_parent_local@4HA":
; CHECK:       .seh_endprologue
; CHECK-NOT:   {{\[(sp|r11)}}
; CHECK:       ldr r0, [r6, #[[#PL]]]
; CHECK-NOT:   {{\[(sp|r11)}}
; CHECK:       str r0, [r6, #[[#PL]]]
; CHECK-NOT:   {{\[(sp|r11)}}
; CHECK:       movw r0
entry:
  %local = alloca i32, align 4
  store volatile i32 %x, ptr %local, align 4
  invoke void @g() to label %done unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  %v = load volatile i32, ptr %local, align 4
  %w = add i32 %v, 1
  store volatile i32 %w, ptr %local, align 4
  catchret from %cp to label %done
done:
  %r = load volatile i32, ptr %local, align 4
  ret i32 %r
}

; A 16-byte vector temporary: the frame is not realigned, so its store has no
; alignment hint.
define void @vectemp() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: vectemp:
; CHECK-NOT:   bfc
; CHECK:       vst1.64 {{.*}}, [r{{[0-9]+}}],
; CHECK:       .seh_endproc
entry:
  %v = load volatile <4 x i32>, ptr @vec, align 16
  %i = load volatile i32, ptr @idx
  %e = extractelement <4 x i32> %v, i32 %i
  store volatile i32 %e, ptr @out
  invoke void @g() to label %cont unwind label %cleanup
cont:
  ret void
cleanup:
  %p = cleanuppad within none []
  call void @dtor(ptr null) [ "funclet"(token %p) ]
  cleanupret from %p unwind to caller
}
