; Ensure we can pass -scalar-evolution-max-iterations multiple times
; RUN: opt -S -passes='require<scalar-evolution>' -scalar-evolution-max-iterations=42 -scalar-evolution-max-iterations=42 < %s
