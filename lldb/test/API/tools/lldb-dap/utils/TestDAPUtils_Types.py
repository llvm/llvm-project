import copy
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

from lldbsuite.test.tools.lldb_dap.dap_types import (
    EmptyBodyResponse,
    Event,
    EventName,
    OutputEvent,
    Request,
    SetBreakpointsArgs,
    Source,
    StackTraceResponse,
    StoppedEvent,
    StoppedReason,
    dict_to_message,
    message_to_dict,
)

T = TypeVar("T")

_Color = Literal["RED", "BLUE", "YELLOW"]
_Number = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_ColorOrNumber = Literal[_Color, _Number]
_ColorNumberOrString = Literal[Literal["ONE", "TWO", Literal["THREE"]], _ColorOrNumber]


class TestDAPUtils_Types(unittest.TestCase):
    """Test serialization and deserialization of different dap types."""

    def verify_round_trip(self, value_type: Type, value: dict):
        message = dict_to_message(value_type, value)
        gotten_value = message_to_dict(message)
        self.assertDictEqual(value, gotten_value)

        # Check the dict values get reconverted to the appropriate types.
        for expected_key, expected_value in gotten_value.items():
            self.assertEqual(type(expected_key), str, "expects type of key to be str.")
            original_value = value[expected_key]
            self.assertEqual(type(expected_value), type(original_value))

    def test_encode_and_decode(self):
        # Self referencing and optional type.
        source_dict = {
            "name": "main.cpp",
            "path": "/some/random/path/to/a/file",
            "sources": [{"name": "main.c", "path": "/another/random/path/to/file"}],
        }
        self.verify_round_trip(Source, source_dict)

        # Argument and nested type.
        source_breakpoint_dict = {
            "source": {
                "name": "CreateASTUnitFromArgs.cpp",
                "path": "/llvm-project/clang/lib/Driver/CreateASTUnitFromArgs.cpp",
            },
            "lines": [86],
            "breakpoints": [{"line": 86}, {"line": 30}, {"line": 100}],
            "sourceModified": False,
        }
        self.verify_round_trip(SetBreakpointsArgs, source_breakpoint_dict)

        # Response type.
        configuration_done_response_dict = {
            "command": "configurationDone",
            "request_seq": 16,
            "seq": 28,
            "success": True,
            "type": "response",
        }
        self.verify_round_trip(EmptyBodyResponse, configuration_done_response_dict)

        stack_trace_dict = {
            "body": {
                "stackFrames": [
                    {
                        "column": 11,
                        "id": 524288,
                        "instructionPointerReference": "0x555555555266",
                        "line": 23,
                        "moduleId": "2833EAD0-0FDC-66C8-88B1-8C6E1D82736C-AE103AE6",
                        "name": "main",
                        "source": {
                            "name": "convert.cpp",
                            "path": "/path/to/where/convert.cpp",
                        },
                    }
                ],
                "totalFrames": 21,
            },
            "command": "stackTrace",
            "request_seq": 21,
            "seq": 40,
            "success": True,
            "type": "response",
        }
        self.verify_round_trip(StackTraceResponse, stack_trace_dict)

        # Event type.
        output_event_dict = {
            "body": {
                "category": "stdout",
                "output": "print malformed utf8 𐌶𐌰L𐌾𐍈 C𐍈𐌼𐌴𐍃 lone trailing �� bytes\r\n",
            },
            "event": "output",
            "seq": 37,
            "type": "event",
        }
        self.verify_round_trip(OutputEvent, output_event_dict)

        stopped_event_dict = {
            "body": {
                "allThreadsStopped": True,
                "description": "breakpoint 3.1",
                "hitBreakpointIds": [3],
                "reason": "breakpoint",
                "text": "breakpoint 3.1",
                "threadId": 4039875,
            },
            "event": "stopped",
            "seq": 38,
            "type": "event",
        }
        self.verify_round_trip(StoppedEvent, stopped_event_dict)

    def test_failing_encode_and_decode(self):
        """Test encoding will fail if the type or value is wrong."""
        # Verify missing field type.
        with self.assertRaises(TypeError):
            configuration_request_dict = {
                "command": "configurationDone",
                "seq": 16,
            }
            self.verify_round_trip(Request, configuration_request_dict)

        # Event with the wrong value for the type key.
        with self.assertRaises(ValueError, msg="expects events with the correct type"):
            process_event_dict = {
                "body": {
                    "isLocalProcess": True,
                    "name": "/path/to/test/build/test",
                    "pointerSize": 64,
                    "startMethod": "launch",
                    "systemProcessId": 4039875,
                },
                "event": "process",
                "seq": 27,
                "type": "response",  # wrong type
            }
            Event.from_json(process_event_dict)

    def test_primitives(self):
        """Test encoding and decoding of python primitives."""

        @dataclass
        class Primitives:
            pbool: bool
            pint: int
            pfloat: float
            pstr: str
            pbytes: bytes

        prim_dict = {
            "pbool": True,
            "pint": 10,
            "pfloat": 3.14,
            "pstr": "HeLL0",
            "pbytes": b"HeLL0 Bytes",
        }
        prim_obj = dict_to_message(Primitives, prim_dict)
        self.assertEqual(prim_obj.pbool, True)
        self.assertEqual(type(prim_obj.pbool), bool)

        self.assertEqual(prim_obj.pint, 10)
        self.assertEqual(type(prim_obj.pint), int)

        self.assertEqual(prim_obj.pfloat, 3.14)
        self.assertEqual(type(prim_obj.pfloat), float)

        self.assertEqual(prim_obj.pstr, "HeLL0")
        self.assertEqual(type(prim_obj.pstr), str)

        self.assertEqual(prim_obj.pbytes, b"HeLL0 Bytes")
        self.assertEqual(type(prim_obj.pbytes), bytes)

        self.verify_round_trip(Primitives, prim_dict)

    def test_literals(self):
        """Test that encoding or decoding dataclass with any form of literal
        is typechecked correctly.
        """

        @dataclass
        class WithLiteral:
            color: _Color
            id: Optional[_Number] = None
            any: Optional[_ColorOrNumber] = None
            color_opt: Optional[_Color] = None
            reason: Optional[StoppedReason] = None

        @dataclass
        class NestedLiterals:
            value: _ColorNumberOrString

        # Literals
        color_dict = {"color": "RED", "id": 2, "color_opt": "BLUE"}
        self.verify_round_trip(WithLiteral, color_dict)

        color_dict_no_opt = {"color": "RED", "id": 1}
        self.verify_round_trip(WithLiteral, color_dict_no_opt)

        literal_dict = {"color": "BLUE", "id": 0, "any": "YELLOW", "reason": "goto"}
        self.verify_round_trip(WithLiteral, literal_dict)
        message = dict_to_message(WithLiteral, literal_dict)
        self.assertEqual(message.color, literal_dict["color"])
        self.assertEqual(message.id, literal_dict["id"])
        self.assertIsNotNone(message.id)
        self.assertEqual(message.any, literal_dict["any"])
        self.assertIsNone(message.color_opt)
        self.assertEqual(message.reason, literal_dict["reason"])

        with self.assertRaises(ValueError):
            # '1' is string instead of int
            mixed_lit_dict = {"color": "BLUE", "id": 7, "any": "1"}
            self.verify_round_trip(WithLiteral, mixed_lit_dict)

        lit_dicts = [
            {"value": 1},
            {"value": "ONE"},
            {"value": "BLUE"},
            {"value": "THREE"},
        ]
        for a_dict in lit_dicts:
            self.verify_round_trip(NestedLiterals, a_dict)

        with self.assertRaises(ValueError):
            a_dict = {"value": 100}  # Not part of the specified literals.
            self.verify_round_trip(NestedLiterals, a_dict)

    def test_unions(self):
        """Test that Union types encodes and decodes correctly"""

        @dataclass
        class WithPrimitiveUnion:
            module_id: Union[int, str]

        obj = dict_to_message(WithPrimitiveUnion, {"module_id": 42})
        self.assertIsInstance(obj.module_id, int)
        self.assertEqual(obj.module_id, 42)

        obj = dict_to_message(WithPrimitiveUnion, {"module_id": "abc-def"})
        self.assertIsInstance(obj.module_id, str)
        self.assertEqual(obj.module_id, "abc-def")

        # Round-trip preserves the correct type.
        self.verify_round_trip(WithPrimitiveUnion, {"module_id": 7})
        self.verify_round_trip(WithPrimitiveUnion, {"module_id": "x"})

        # Incompatible data raises immediately with a clear error.
        with self.assertRaises(TypeError) as ctx:
            dict_to_message(WithPrimitiveUnion, {"module_id": [1, 2, 3]})
        self.assertIn("is compatible with", str(ctx.exception))

        # Union of list and dict.
        @dataclass
        class WithListOrDict:
            source_map: Union[List[Tuple[str, str]], Dict[str, str]]

        obj_list = dict_to_message(
            WithListOrDict, {"source_map": [["a", "b"], ["c", "d"]]}
        )
        self.assertIsInstance(obj_list.source_map, list)
        self.assertEqual(
            obj_list.source_map, [("a", "b"), ("c", "d")], "expect list of tuples"
        )

        obj_dict = dict_to_message(WithListOrDict, {"source_map": {"a": "b", "c": "d"}})
        self.assertIsInstance(obj_dict.source_map, dict)
        self.assertEqual(obj_dict.source_map, {"a": "b", "c": "d"})

        # Union of two dataclasses.
        @dataclass
        class TypeA:
            x: int

        @dataclass
        class TypeB:
            y: str

        @dataclass
        class WithDataclassUnion:
            value: Union[TypeA, TypeB]

        obj_a = dict_to_message(WithDataclassUnion, {"value": {"x": 1}})
        self.assertIsInstance(obj_a.value, TypeA)
        self.assertEqual(obj_a.value.x, 1)  # type: ignore

        obj_b = dict_to_message(WithDataclassUnion, {"value": {"y": "hello"}})
        self.assertIsInstance(obj_b.value, TypeB)
        self.assertEqual(obj_b.value.y, "hello")  # type: ignore

        # When only one Union type matches, the error should contain that type.
        # In this case only 'RequiresX' will match 999.
        @dataclass
        class RequiresX:
            x: int

        @dataclass
        class WithSingleCompatible:
            value: Union[List[int], str, RequiresX]

        with self.assertRaises(TypeError) as ctx:
            dict_to_message(WithSingleCompatible, {"value": {"z": 999}})
        self.assertIn("RequiresX", str(ctx.exception))

        # Nested Optional value.
        @dataclass
        class WithOptionalUnion:
            value: Optional[Union[int, str]] = None

        obj_none = dict_to_message(WithOptionalUnion, {})
        self.assertIsNone(obj_none.value)

        obj_none = dict_to_message(WithOptionalUnion, {"value": None})
        self.assertIsNone(obj_none.value)

        obj_int = dict_to_message(WithOptionalUnion, {"value": 5})
        self.assertEqual(obj_int.value, 5)

        obj_str = dict_to_message(WithOptionalUnion, {"value": "hi"})
        self.assertEqual(obj_str.value, "hi")

        with self.assertRaises(TypeError):
            # bool not accepted as int, even though it is a base class.
            dict_to_message(WithOptionalUnion, {"value": True})

    def test_field_alias(self):
        """Test that renaming in a DAP field serializes and deserializes correctly"""

        @dataclass
        class AClass:
            version: str = field(metadata={"alias": "$__lldb_version"})
            color: Optional[_Color] = field(metadata={"alias": "colour"})

        a_dict = {"$__lldb_version": "20.20.20", "colour": "RED"}
        dict_obj = dict_to_message(AClass, a_dict)
        self.assertEqual(dict_obj.version, a_dict["$__lldb_version"])
        self.assertEqual(dict_obj.color, a_dict["colour"])

        self.assertEqual(a_dict, message_to_dict(dict_obj))

    def test_required_field(self):
        """Test that fields with required metadata enforce their value on decode."""

        @dataclass
        class WithRequired:
            color: str = field(metadata={"required": "RED"})
            name: str = ""

        # Test present and correct value.
        obj = dict_to_message(WithRequired, {"color": "RED", "name": "test"})
        self.assertEqual(obj.color, "RED")

        # Test missing key.
        with self.assertRaises(TypeError):
            dict_to_message(WithRequired, {"name": "test"})

        # Test wrong value raises ValueError.
        with self.assertRaises(ValueError):
            dict_to_message(WithRequired, {"color": "BLUE", "name": "test"})

        result = message_to_dict(obj)
        self.assertEqual(result["color"], "RED")

        # Test non-string required values.
        @dataclass
        class WithRequiredInt:
            version: int = field(metadata={"required": 2})

        obj_int = dict_to_message(WithRequiredInt, {"version": 2})
        self.assertEqual(obj_int.version, 2)

        with self.assertRaises(TypeError):
            dict_to_message(WithRequiredInt, {})

        with self.assertRaises(ValueError):
            dict_to_message(WithRequiredInt, {"version": 3})

        # Test 'required' and 'alias' metadata can be combined.
        @dataclass
        class WithRequiredAlias:
            kind: str = field(metadata={"required": "PURPLE", "alias": "type"})

        obj_alias = dict_to_message(WithRequiredAlias, {"type": "PURPLE"})
        self.assertEqual(obj_alias.kind, "PURPLE")

        with self.assertRaises(ValueError):
            dict_to_message(WithRequiredAlias, {"type": "response"})

    def test_event_field_constant(self):
        """Test that event subclasses always have the correct event name set."""
        stopped_dict = {
            "body": {
                "reason": "breakpoint",
                "threadId": 1,
                "allThreadsStopped": True,
            },
            "event": "stopped",
            "seq": 10,
            "type": "event",
        }
        event = dict_to_message(StoppedEvent, stopped_dict)
        self.assertEqual(event.event, EventName.STOPPED)

        # Serialization round-trips back to the correct event name.
        result = message_to_dict(event)
        self.assertEqual(result["event"], "stopped")

        # Missing event key.
        with self.assertRaises(TypeError):
            without_event = {k: v for k, v in stopped_dict.items() if k != "event"}
            dict_to_message(StoppedEvent, without_event)

        # Test wrong event type.
        with self.assertRaises(ValueError):
            dict_to_message(StoppedEvent, dict(stopped_dict, event="output"))

    def test_dictionary(self):
        """Tests DAP types with dictionary decodes and encodes correctly"""

        @dataclass
        class MessageDict:
            data: Dict[str, Union[str, None]]
            data_opt: Optional[Dict[str, Optional[str]]] = None

        a_dict = {
            "data": {"FOO": None, "NO_COLOR": "TRUE", "OTHER": ""},
            "data_opt": None,
        }
        dict_obj = dict_to_message(MessageDict, a_dict)
        self.assertIsNotNone(dict_obj.data)
        dict_obj_data = cast(dict, dict_obj.data)
        self.assertEqual(dict_obj_data["FOO"], a_dict["data"]["FOO"])
        self.assertEqual(dict_obj_data["NO_COLOR"], a_dict["data"]["NO_COLOR"])
        self.assertEqual(dict_obj_data["OTHER"], a_dict["data"]["OTHER"])
        self.assertEqual(dict_obj.data_opt, a_dict["data_opt"])

        # We skip fields that is None. when converting dataclass to dict.
        without_data_opt = copy.deepcopy(a_dict)
        without_data_opt.pop("data_opt")
        self.assertEqual(without_data_opt, message_to_dict(dict_obj))
