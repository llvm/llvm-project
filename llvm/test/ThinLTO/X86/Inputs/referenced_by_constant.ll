
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @referencedbyglobal() {
    ret void
}

@someglobal = internal unnamed_addr constant i8* bitcast (void ()* @referencedbyglobal to i8*)
@ptr = global i8** null

define  void @bar() #0 align 2 {
  store i8** getelementptr inbounds (i8*, i8** @someglobal, i64 0) , i8*** @ptr, align 8
  ret void
}

