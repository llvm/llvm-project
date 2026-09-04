
// LLDB C++ API Test: Evaluate an expression that yields a global struct from
// several threads at the same time and inspect the resulting SBValue and
// SBType from each of these threads.
//
// All threads debug the same program (and therefore share LLDB's module for
// it), and every thread also launches its own process.

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBLaunchInfo.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBTypeEnumMember.h"
#include "lldb/API/SBValue.h"

#include "common.h"

using namespace lldb;

namespace {

/// The number of threads used to debug the shared program at the same time.
const unsigned num_threads = 5;

/// How often each thread evaluates and inspects the expression result.
const unsigned num_iterations = 5;

/// Blocks until all threads have arrived.
class Barrier {
  std::mutex m_mutex;
  std::condition_variable m_condition;
  unsigned m_missing;

public:
  Barrier(unsigned count) : m_missing(count) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_missing--;
    if (m_missing == 0)
      m_condition.notify_all();
    else
      m_condition.wait(lock, [this] { return m_missing == 0; });
  }
};

/// Returns the value of the given SBValue as a signed integer.
int64_t get_signed(SBValue value, const std::string &what) {
  expect(value.IsValid(), what + ": value is invalid");
  SBError error;
  int64_t result = value.GetValueAsSigned(error, 0);
  if (error.Fail())
    throw Exception(what + ": " + error.GetCString());
  return result;
}

/// Calls various SBType functions on the type of the global 'g_data' struct.
void check_struct_type(SBType type) {
  expect(type.IsValid(), "type of 'g_data' is invalid");
  expect_string(type.GetName(), "Data", "SBType::GetName");
  expect_string(type.GetDisplayTypeName(), "Data",
                "SBType::GetDisplayTypeName");
  expect(type.GetTypeClass() == eTypeClassStruct, "SBType::GetTypeClass");
  expect(type.IsAggregateType(), "'Data' should be an aggregate type");
  expect(type.IsTypeComplete(), "'Data' should be a complete type");
  expect(!type.IsPointerType(), "'Data' should not be a pointer type");
  expect(type.GetByteSize() > 0, "'Data' has no byte size");
  expect(type.GetByteAlign() > 0, "'Data' has no alignment");
  expect_int(type.GetNumberOfTemplateArguments(), 0,
             "SBType::GetNumberOfTemplateArguments");

  // Check the fields of the struct.
  expect_int(type.GetNumberOfFields(), 5, "SBType::GetNumberOfFields");
  SBTypeMember first_field = type.GetFieldAtIndex(0);
  expect(first_field.IsValid(), "first field of 'Data' is invalid");
  expect_string(first_field.GetName(), "i", "name of first field");
  expect_string(first_field.GetType().GetName(), "int",
                "type of first field 'i'");

  SBTypeMember array_field = type.GetFieldAtIndex(3);
  expect_string(array_field.GetName(), "array", "name of fourth field");
  SBType array_type = array_field.GetType();
  expect(array_type.IsArrayType(), "'array' should be an array type");
  expect_string(array_type.GetArrayElementType().GetName(), "int",
                "element type of 'array'");

  // Check the base class and the nested type.
  expect_int(type.GetNumberOfDirectBaseClasses(), 1,
             "SBType::GetNumberOfDirectBaseClasses");
  expect_string(type.GetDirectBaseClassAtIndex(0).GetName(), "Base",
                "name of direct base class");
  expect_int(type.GetNumberOfVirtualBaseClasses(), 0,
             "SBType::GetNumberOfVirtualBaseClasses");
  SBType nested_type = type.FindDirectNestedType("Inner");
  expect(nested_type.IsValid(), "'Data' has no nested type 'Inner'");
  expect_int(nested_type.GetNumberOfFields(), 1,
             "number of fields of 'Data::Inner'");

  // Check the static member.
  SBTypeStaticField static_field = type.GetStaticFieldWithName("static_field");
  expect(static_field.IsValid(), "'Data' has no static field 'static_field'");
  expect_string(static_field.GetName(), "static_field",
                "SBTypeStaticField::GetName");
  expect_string(static_field.GetType().GetName(), "int",
                "type of 'Data::static_field'");

  // Check the member function of the struct.
  bool found_getter = false;
  for (uint32_t i = 0; i < type.GetNumberOfMemberFunctions(); ++i) {
    SBTypeMemberFunction func = type.GetMemberFunctionAtIndex(i);
    expect(func.IsValid(), "invalid member function");
    const char *func_name = func.GetName();
    expect(func_name != nullptr, "member function without a name");
    if (std::strcmp(func_name, "GetI") != 0)
      continue;
    found_getter = true;
    expect(func.GetKind() == eMemberFunctionKindInstanceMethod,
           "'GetI' should be an instance method");
    expect_string(func.GetReturnType().GetName(), "int",
                  "return type of 'GetI'");
    expect_int(func.GetNumberOfArguments(), 0, "number of arguments of 'GetI'");
  }
  expect(found_getter, "'Data' has no member function 'GetI'");

  // Check creating derived types.
  // This should never race as the types are supposed to be created in the
  // scratch type system.
  SBType pointer_type = type.GetPointerType();
  expect(pointer_type.IsPointerType(), "SBType::GetPointerType");
  SBType pointee_type = pointer_type.GetPointeeType();
  expect(pointee_type == type, "pointee type of 'Data *' is not 'Data'");
  SBType reference_type = type.GetReferenceType();
  expect(reference_type.IsReferenceType(), "SBType::GetReferenceType");
  SBType dereferenced_type = reference_type.GetDereferencedType();
  expect(dereferenced_type == type,
         "dereferenced type of 'Data &' is not 'Data'");
  SBType data_array_type = type.GetArrayType(4);
  expect(data_array_type.IsArrayType(), "SBType::GetArrayType");
  expect_int(data_array_type.GetByteSize(), 4 * type.GetByteSize(),
             "byte size of 'Data[4]'");
  expect_string(type.GetCanonicalType().GetName(), "Data",
                "SBType::GetCanonicalType");
  expect(type.GetUnqualifiedType().IsValid(), "SBType::GetUnqualifiedType");
  expect_string(type.GetBasicType(eBasicTypeInt).GetName(), "int",
                "SBType::GetBasicType");

  SBStream description;
  expect(type.GetDescription(description, eDescriptionLevelFull),
         "SBType::GetDescription failed");
  expect(description.GetSize() > 0, "SBType::GetDescription gave no output");
}

/// Calls various SBValue functions on the value of the global 'g_data' struct.
void check_struct_value(SBValue value, SBTarget &target) {
  expect(value.IsValid(), "expression result is invalid");
  SBError error = value.GetError();
  if (error.Fail())
    throw Exception(std::string("expression failed: ") + error.GetCString());

  expect(value.GetName() != nullptr, "expression result has no name");
  expect_string(value.GetTypeName(), "Data", "SBValue::GetTypeName");
  expect_string(value.GetDisplayTypeName(), "Data",
                "SBValue::GetDisplayTypeName");
  expect_int(value.GetByteSize(), value.GetType().GetByteSize(),
             "byte size of value and its type disagree");
  expect(value.GetValueType() != eValueTypeInvalid, "SBValue::GetValueType");
  expect(value.MightHaveChildren(), "a struct should have children");
  expect(!value.IsSynthetic(), "'g_data' should not have synthetic children");
  expect(value.GetTarget() == target, "SBValue::GetTarget");

  // The base class and the five members of the struct.
  expect_int(value.GetNumChildren(), 6, "SBValue::GetNumChildren");

  SBValue base = value.GetChildAtIndex(0);
  expect_string(base.GetName(), "Base", "name of the base class child");
  expect_int(get_signed(base.GetChildMemberWithName("base_field"),
                        "g_data.base_field"),
             1, "value of 'g_data.base_field'");

  SBValue i = value.GetChildMemberWithName("i");
  expect(i.IsValid(), "'g_data' has no child 'i'");
  expect_string(i.GetTypeName(), "int", "type of 'g_data.i'");
  expect_string(i.GetValue(), "42", "SBValue::GetValue for 'g_data.i'");
  expect_int(get_signed(i, "g_data.i"), 42, "value of 'g_data.i'");
  expect_int(value.GetIndexOfChildWithName("i"), 1,
             "SBValue::GetIndexOfChildWithName");
  expect(i.AddressOf().IsValid(), "SBValue::AddressOf for 'g_data.i'");
  expect(i.GetParent().IsValid(), "SBValue::GetParent for 'g_data.i'");

  SBValue inner_field = value.GetValueForExpressionPath(".inner.inner_field");
  expect_int(get_signed(inner_field, "g_data.inner.inner_field"), 20,
             "value of 'g_data.inner.inner_field'");

  SBValue color = value.GetChildMemberWithName("color");
  expect_string(color.GetValue(), "eGreen", "value of 'g_data.color'");
  expect_int(get_signed(color, "g_data.color"), 2,
             "integer value of 'g_data.color'");
  expect_int(color.GetType().GetEnumMembers().GetSize(), 3,
             "number of enumerators of 'Color'");

  SBValue array = value.GetChildMemberWithName("array");
  expect_int(array.GetNumChildren(), 3, "number of children of 'g_data.array'");
  expect_int(get_signed(array.GetChildAtIndex(2), "g_data.array[2]"), 3,
             "value of 'g_data.array[2]'");

  SBValue str = value.GetChildMemberWithName("str");
  expect(str.GetType().IsPointerType(), "'g_data.str' should be a pointer");
  expect_string(str.GetTypeName(), "const char *", "type of 'g_data.str'");
  expect(str.GetValueAsUnsigned(0) != 0, "'g_data.str' should not be null");
  expect_string(str.GetType().GetPointeeType().GetName(), "const char",
                "pointee type of 'g_data.str'");

  SBData data = value.GetData();
  expect(data.IsValid(), "SBValue::GetData");
  expect_int(data.GetByteSize(), value.GetByteSize(),
             "byte size of the data of 'g_data'");

  SBStream description;
  expect(value.GetDescription(description), "SBValue::GetDescription failed");
  const char *description_text = description.GetData();
  expect(description_text && std::strstr(description_text, "i = 42"),
         "SBValue::GetDescription doesn't mention the value of 'i'");

  expect(value.GetProcess().IsValid(), "SBValue::GetProcess");
  expect(i.GetLoadAddress() != LLDB_INVALID_ADDRESS,
         "'g_data.i' has no load address");
}

/// Creates a target for the given program, evaluates an expression for the
/// global 'g_data' struct and then inspects the result. Returns an empty
/// string on success or otherwise a description of the encountered failure.
std::string evaluate_and_check(SBDebugger &dbg, const std::string &program,
                               Barrier &barrier) {
  SBTarget target;
  SBProcess process;
  std::string setup_error;
  try {
    target = dbg.CreateTarget(program.c_str());
    expect(target.IsValid(), "invalid target");

    SBBreakpoint bkpt = target.BreakpointCreateByName("main");
    expect(bkpt.GetNumLocations() > 0, "breakpoint got no locations");

    SBError error;
    SBLaunchInfo launch_info = target.GetLaunchInfo();
    process = target.Launch(launch_info, error);
    if (error.Fail())
      throw Exception(std::string("failed to launch process: ") +
                      error.GetCString());
    expect(process.GetState() == eStateStopped, "process was not stopped");
  } catch (Exception &e) {
    setup_error = e.what();
  }

  // Wait for the other threads so that the expressions below are evaluated at
  // the same time. This also has to happen when the setup above failed as the
  // other threads would otherwise wait forever.
  barrier.Wait();

  std::string error_message = setup_error;
  if (error_message.empty()) {
    try {
      for (unsigned i = 0; i < num_iterations; ++i) {
        SBValue value = target.EvaluateExpression("g_data");
        check_struct_value(value, target);
        check_struct_type(value.GetType());
      }
    } catch (Exception &e) {
      error_message = e.what();
    }
  }

  if (process.IsValid())
    process.Kill();
  return error_message;
}

} // namespace

void test(SBDebugger &dbg, std::vector<std::string> args) {
  dbg.SetAsync(false);

  Barrier barrier(num_threads);
  // Every thread reports its failure (if any) in its own slot.
  std::vector<std::string> errors(num_threads);
  std::vector<std::thread> threads;

  for (unsigned i = 0; i < num_threads; ++i)
    threads.emplace_back(
        [&, i]() { errors[i] = evaluate_and_check(dbg, args.at(0), barrier); });
  for (std::thread &thread : threads)
    thread.join();

  for (unsigned i = 0; i < num_threads; ++i)
    if (!errors[i].empty())
      throw Exception("thread " + std::to_string(i) + ": " + errors[i]);
}
