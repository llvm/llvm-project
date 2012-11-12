; No arguments means internalize everything
; RUN: opt < %s -internalize -S | FileCheck --check-prefix=NOARGS %s

; Internalize all but foo and j
; RUN: opt < %s -internalize -internalize-public-api-list foo -internalize-public-api-list j -S | FileCheck --check-prefix=LIST %s

; Non existent files should be treated as if they were empty (so internalize
; everything)
; RUN: opt < %s -internalize -internalize-public-api-file /nonexistent/file 2> /dev/null -S | FileCheck --check-prefix=EMPTYFILE %s

; RUN: opt < %s -S -internalize -internalize-public-api-list bar -internalize-public-api-list foo -internalize-public-api-file /nonexistent/file  2> /dev/null | FileCheck --check-prefix=LIST2 %s

; -file and -list options should be merged, the .apifile contains foo and j
; RUN: opt < %s -internalize -internalize-public-api-list bar -internalize-public-api-file %s.apifile -S | FileCheck --check-prefix=MERGE %s

; NOARGS: @i = internal global
; LIST: @i = internal global
; EMPTYFILE: @i = internal global
; LIST2: @i = internal global
; MERGE: @i = internal global
@i = global i32 0

; NOARGS: @j = internal global
; LIST: @j = global
; EMPTYFILE: @j = internal global
; LIST2: @j = internal global
; MERGE: @j = global
@j = global i32 0

; NOARGS: define internal void @main
; LIST: define internal void @main
; EMPTYFILE: define internal void @main
; LIST2: define internal void @main
; MERGE: define internal void @main
define void @main() {
        ret void
}

; NOARGS: define internal void @foo
; LIST: define void @foo
; EMPTYFILE: define internal void @foo
; LIST2: define void @foo
; MERGE: define void @foo
define void @foo() {
        ret void
}

; NOARGS: define internal void @bar
; LIST: define internal void @bar
; EMPTYFILE: define internal void @bar
; LIST2: define void @bar
; MERGE: define void @bar
define void @bar() {
        ret void
}
