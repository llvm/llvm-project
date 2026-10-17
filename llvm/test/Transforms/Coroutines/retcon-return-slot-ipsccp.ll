; RUN: opt -verify-each -passes='ipsccp' -S %s | FileCheck %s --check-prefix=PRE
; RUN: opt -verify-each -passes='cgscc(function-attrs),ipsccp' -S %s | FileCheck %s --check-prefix=PRE
; RUN: opt -verify-each -passes='thinlto<O2>' -S %s | FileCheck %s --check-prefix=POST

; An ICP-style direct/indirect call join must retain the direct return pair.
; The return is represented explicitly, without relying on auto-upgrade.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare ptr @llvm.coro.prepare.retcon(ptr)
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @resume(ptr, i1)
declare void @consume(ptr)

define internal swiftcc { ptr, ptr } @accessor(ptr noalias %buffer,
                                                ptr swiftself %object) #0 {
entry:
  %coro.ret = alloca [16 x i8], align 8
  %id = call token @llvm.coro.id.retcon.once(
      i32 32, i32 8, ptr %buffer, ptr @resume, ptr @malloc, ptr @free, ptr %coro.ret)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  %field = getelementptr i8, ptr %object, i64 8
  %suspended = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %field)
  call void @llvm.coro.end(ptr %frame, i1 false, token none)
  %coro.ret.load = load { ptr, ptr }, ptr %coro.ret
  ret { ptr, ptr } %coro.ret.load
}

define void @caller(ptr %target, ptr %buffer, ptr %object) {
entry:
  %prepared = call ptr @llvm.coro.prepare.retcon(ptr %target)
  %is_direct = icmp eq ptr %prepared, @accessor
  br i1 %is_direct, label %direct, label %indirect

direct:
  %direct_pair = call swiftcc { ptr, ptr } @accessor(
      ptr noalias %buffer, ptr swiftself %object)
  br label %join

indirect:
  %indirect_pair = call swiftcc { ptr, ptr } %prepared(
      ptr noalias %buffer, ptr swiftself %object)
  br label %join

join:
  %pair = phi { ptr, ptr } [ %direct_pair, %direct ],
                          [ %indirect_pair, %indirect ]
  %continuation = extractvalue { ptr, ptr } %pair, 0
  %field = extractvalue { ptr, ptr } %pair, 1
  call void @consume(ptr %field)
  call swiftcc void %continuation(ptr %buffer, i1 false)
  ret void
}

; PRE-LABEL: define internal swiftcc { ptr, ptr } @accessor(
; PRE: %coro.ret = alloca [16 x i8], align 8
; PRE: call token @llvm.coro.id.retcon.once({{.*}}ptr %coro.ret)
; PRE: load { ptr, ptr }, ptr %coro.ret
; PRE: ret { ptr, ptr }
; PRE-LABEL: define void @caller(
; PRE: %direct_pair = call swiftcc { ptr, ptr } @accessor(ptr noalias %buffer, ptr swiftself %object){{$}}
; PRE: %pair = phi { ptr, ptr } [ %direct_pair, %direct ], [ %indirect_pair, %indirect ]
; PRE-NOT: noreturn

; POST-LABEL: define internal swiftcc { ptr, ptr } @accessor(
; POST: ret { ptr, ptr }
; POST-LABEL: define void @caller(
; POST: %pair = phi { ptr, ptr } [ %direct_pair, %direct ], [ %indirect_pair, %indirect ]

attributes #0 = { noinline presplitcoroutine }
