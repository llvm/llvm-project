#pragma once

#include <cstdint>
#include <string_view>

enum class NSAPICommandCategory {
  Common,
  OpenMP,
  Process,
  Validation,
};

enum NSAPICommandOpcode {
  // OpenMP commands
  NSAPI_OMP_SET_NUM_THREADS = 0,
  NSAPI_OMP_SET_DYNAMIC,
  /*...*/
  NSAPI_OMP_MAX = 0x100,
  // Multiprocess commands
  NSAPI_MULTIPROCESS_SET = NSAPI_OMP_MAX + 1,
  /*...*/
  NSAPI_MULTIPROCESS_MAX = 0x200,

  // Maximal value for NSAPI opcode in the LLVM side - nextutils opcodes will
  // start after this
  NSAPI_LLVM_MAX = 0x1000,
};

typedef uint16_t NSAPIOpcode;

struct NSAPICommand {
private:
  NSAPIOpcode _opcode;

protected:
  explicit NSAPICommand(uint32_t opcode) : _opcode(opcode) {}

public:
  NSAPIOpcode Opcode() { return _opcode; }
};

class NSAPIHandler {
private:
  NSAPIHandler(const NSAPIHandler &) = delete;
  NSAPIHandler(NSAPIHandler &&) = delete;
  NSAPIHandler &operator=(const NSAPIHandler &) = delete;
  NSAPIHandler &operator=(const NSAPIHandler &&) = delete;

  static NSAPIHandler *_current;

protected:
  explicit NSAPIHandler() = default;
  virtual ~NSAPIHandler() = default;

public:
  virtual int Execute(NSAPICommand &cmd) = 0;

  static void Register(NSAPIHandler *handler) { _current = handler; }
  static NSAPIHandler &Current() { return *_current; }
};

extern "C" {
NSAPIHandler *nsapi_get_current_handler();
}
