; RUN: llvm-as < %s | llc -march=c

        %JNIEnv = type %struct.JNINa*
        %struct.JNINa = type { i8*, i8*, i8*, void (%JNIEnv*)* }

