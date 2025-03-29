; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

%A = type { double }
; the intermediary struct causes A_arr and B_arr to be different types
%A_struct = type { %A }
%A_arr = type { [1 x %A_struct] }

%B = type { double }
%B_struct = type { %B }
%B_arr = type { [1 x %B_struct] }

declare void @noop()

define %A_arr @a() {
; CHECK-LABEL: define %A_arr @a() {
; CHECK-NEXT:    call void @noop()
; CHECK-NEXT:    ret %A_arr zeroinitializer
;
  call void @noop()
  ret %A_arr zeroinitializer
}

define %B_arr @b() {
; CHECK-LABEL: define %B_arr @b() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call %A_arr @a
; CHECK-NEXT:    [[TMP2:%.*]] = extractvalue %A_arr [[TMP1]], 0
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue [1 x %A_struct] [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue %A_struct [[TMP3]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue %A [[TMP4]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue %B poison, double [[TMP5]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = insertvalue %B_struct poison, %B [[TMP6]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = insertvalue [1 x %B_struct] poison, %B_struct [[TMP7]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = insertvalue %B_arr poison, [1 x %B_struct] [[TMP8]], 0
; CHECK-NEXT:    ret %B_arr [[TMP9]]
;
  call void @noop()
  ret %B_arr zeroinitializer
}
