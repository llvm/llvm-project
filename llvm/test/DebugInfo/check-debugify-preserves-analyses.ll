; RUN: opt < %s -globals-aa -functionattrs | \
; RUN:   opt -S -strip -strip-dead-prototypes -strip-named-metadata > %t.no_dbg

; RUN: opt < %s -debugify-each -globals-aa -functionattrs | \
; RUN:   opt -S -strip -strip-dead-prototypes -strip-named-metadata > %t.with_dbg

; RUN: diff %t.no_dbg %t.with_dbg

define i32 @f_1(i32 %x) {
  %tmp = call i32 @f_1(i32 0) [ "deopt"() ]
  ret i32 0
}
