; RUN: not opt < %s 2>&1 | grep 'not a number, or does not fit in an unsigned int'

target datalayout = "p:4294967296:64:64"
