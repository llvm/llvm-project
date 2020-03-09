// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
class A
{
    var x : Int64
    var y : Float

    init (int : Int64)
    {
        x = int
        y = Float(int) + 0.5
    }
    init (float : Float)
    {
        x = Int64(float)
        y = float
    }

    func do_something (_ input: Int64) -> Int64
    {
        return x * input;
    }
}

struct B
{
    var b_int : Int
    var b_read_only : Int
    {
        return 5
    }
}

extension B
{
    var b_float : Float
    {
        return Float(b_int) + 0.5
    }
}

enum SomeValues : Equatable 
{
    case Five 
    case Six 
    case Eleven 
}

func == (lhs: SomeValues, rhs: SomeValues) -> Bool
{
    switch (lhs, rhs)
    {
        case (.Five, .Five):
            fallthrough
        case (.Six, .Six):
            fallthrough
        case (.Eleven, .Eleven):
            return true
        case (_, _):
            return false
    }
}

extension SomeValues 
{
    func toInt() -> Int
    {
        switch self
        {
        case .Five:
            return 5
        case .Six:
            return 6
        case .Eleven:
            return 11
        }
    }
}
  
func main () -> Void
{
    var is_five : Int64 = 5
    var is_six : Int64 = 6
    var is_eleven : Int64 = 11

    var float_5 : Float = 5.0
    var float_6 : Float = 6.0
    var float_11 : Float = 11.0

    var a_obj = A(int: is_six)
    var a_nother_obj = A(float: 6.5)

    var str_int_dict = ["five" : is_five, "six" : is_six, "eleven" : is_eleven]
    var int_str_dict : Dictionary<Int,String> = [5 : "five", 6 : "six", 11 : "eleven"]

    var eleven_by_dict = str_int_dict["five"]! + str_int_dict["six"]! // Dict breakpoint
    var eleven_str = int_str_dict[11]

    var an_int_array : Array<Int64> = [is_five, is_six, is_eleven]
    an_int_array.append(13)

    var b_struct = B(b_int:5)
    b_struct.b_int = b_struct.b_read_only
    var b_struct_2 = B(b_int:20)

    var a_tuple = ( is_five, is_six, is_five + is_six == is_eleven)
    var also_eleven = a_tuple.0 + a_tuple.1

    var enum_eleven = SomeValues.Eleven
    var enum_conversion : Int = SomeValues.Five.toInt() + SomeValues.Six.toInt()
    var enum_math = SomeValues.Eleven.toInt() == enum_conversion

    //var a_char = 'a'
    //var euro_char = '\u20ac'
    //var a_string = "Euro: \u20ac"
    //var euro_is_euro : Bool = a_string[advance(a_string.startIndex,6)] == euro_char

    var €_varname = Int64(5)
    var same_five = €_varname == is_five 

    var int_result = a_obj.x + is_six // Set breakpoint here

    var math_works = is_eleven == is_five + is_six

}

var my_class = A (int : 5)
var my_global : Int = 30

print(my_global)

main()
