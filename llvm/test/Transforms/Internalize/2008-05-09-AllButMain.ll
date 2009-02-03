; No arguments means internalize all but main
; RUN: llvm-as < %s | opt -internalize | llvm-dis | grep internal | count 4
; Internalize all but foo and j
; RUN: llvm-as < %s | opt -internalize -internalize-public-api-list foo -internalize-public-api-list j | llvm-dis | grep internal | count 3
; Non existent files should be treated as if they were empty (so internalize all but main)
; RUN: llvm-as < %s | opt -internalize -internalize-public-api-file /nonexistent/file 2> /dev/null | llvm-dis | grep internal | count 4
; RUN: llvm-as < %s | opt -internalize -internalize-public-api-list bar -internalize-public-api-list foo -internalize-public-api-file /nonexistent/file 2> /dev/null | llvm-dis | grep internal | count 3
; -file and -list options should be merged, the .apifile contains foo and j
; RUN: llvm-as < %s | opt -internalize -internalize-public-api-list bar -internalize-public-api-file %s.apifile | llvm-dis | grep internal | count 2

@i = weak global i32 0          ; <i32*> [#uses=0]
@j = weak global i32 0          ; <i32*> [#uses=0]

define void @main(...) {
entry:  
        ret void
}

define void @foo(...) {
entry:  
        ret void
}

define void @bar(...) {
entry:  
        ret void
}
