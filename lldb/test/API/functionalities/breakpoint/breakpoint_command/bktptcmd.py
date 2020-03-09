from __future__ import print_function
import side_effect

def useless_function(first, second):
    print("I have the wrong number of arguments.")

def function(frame, bp_loc, dict):
    side_effect.bktptcmd = "function was here"

def another_function(frame, bp_loc, extra_args, dict):
    se_value = extra_args.GetValueForKey("side_effect")
    se_string = se_value.GetStringValue(100)
    side_effect.fancy = se_string

def a_third_function(frame, bp_loc, extra_args, dict):
    se_value = extra_args.GetValueForKey("side_effect")
    se_string = se_value.GetStringValue(100)
    side_effect.fancier = se_string

def empty_extra_args(frame, bp_loc, extra_args, dict):
    if extra_args.IsValid():
        side_effect.not_so_fancy = "Extra args should not be valid"
    side_effect.not_so_fancy = "Not so fancy"
