; RUN: llc -verify-machineinstrs < %s | grep 'foo bar":'

target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu.2.0"
@"foo bar" = global i32 4               ; <i32*> [#uses=0]

