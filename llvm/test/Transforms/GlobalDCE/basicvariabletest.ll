; RUN: opt < %s -globaldce -S | not grep global

@X = external global i32
@Y = internal global i32 7

