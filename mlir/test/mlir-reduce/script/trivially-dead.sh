#!/bin/sh

# break only on `arith.constat 2 : i32`
! cat $1 | grep "arith.constant 2 : i32" 2>&1 1>/dev/null
