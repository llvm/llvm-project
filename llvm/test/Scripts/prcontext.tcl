#!/usr/bin/tclsh
#
# Usage:
# prcontext <pattern> <# lines of context>
# (for platforms that don't have grep -C)


#
# Get the arguments
#
set pattern [lindex $argv 0]
set num [lindex $argv 1]


#
# Get all of the lines in the file.
#
set lines [split [read stdin] \n]

set index 0
foreach line $lines {
    if { [regexp $pattern $line match matchline] } {
        if { [ expr [expr $index - $num] < 0 ] } {
            set bottom 0
        } else {
            set bottom [expr $index - $num]
        }
        set endLineNum [ expr [expr $index + $num] + 1]
        while {$bottom < $endLineNum} {
            set output [lindex $lines $bottom]
            puts $output
            set bottom [expr $bottom + 1]
        }
    }
    set index [expr $index + 1]
}