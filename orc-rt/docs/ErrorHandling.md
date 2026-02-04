# ORC-RT Error Handling Policy

## Overview

ORC-RT uses a structured error handling system based on the `orc_rt::Error` and
`orc_rt::Expected<T>` classes. This system provides type-safe error propagation
that works consistently across different compilation configurations (with or
without C++ exceptions).

## Fundamental Principles

### 1. Error Representation
- **Success**: Represented by `Error::success()` - a lightweight, zero-cost value
- **Failure**: Represented by `Error` objects containing typed error information
- **Values with Potential Errors**: Use `Expected<T>` to combine success values
  with error handling

### 2. Error Categories

**Recoverable Errors**: Environmental issues that can be handled gracefully
- File I/O failures, network issues, malformed input
- Use `Error` and `Expected<T>` return types
- Examples: `StringError`, `MyCustomError`

**Programmatic Errors**: Violations of API contracts or program invariants
- Use assertions
- Should terminate the program immediately
- Examples: Unexpected null pointers, invalid enum values

> **Important: Library Design Principles**
>
> **ORC-RT is a library and must never call terminating functions** like `exit()`,
> `abort()`, or `std::terminate()` in response to recoverable errors. Libraries
> should always return errors to their callers, allowing the application to decide
> how to handle them.

## Core Error Types

### Error
```cpp
namespace orc_rt {
  class Error {
  public:
    // Create success value
    static Error success();

    // Check for failure
    explicit operator bool();  // true = failure, false = success

    // Type checking
    template<typename ErrT> bool isA() const;

    // Exception interop (when exceptions enabled)
    void throwOnFailure();
  };
}
```

### Expected<T>
```cpp
template<typename T>
class Expected {
public:
  // Construction
  Expected(T Value);
  Expected(Error Err);

  // Check for success
  explicit operator bool();  // true = success, false = failure

  // Access value (success case)
  T& operator*();
  T* operator->();

  // Extract error (failure case)
  Error takeError();
};
```

## Defining Custom Error Types

Use `ErrorExtends<ThisT, ParentT>`:

```cpp
class CustomError : public ErrorExtends<CustomError, ErrorInfoBase> {
public:
  CustomError(std::string Message) : Message(std::move(Message)) {}

  std::string toString() const noexcept override {
    return "CustomError: " + Message;
  }

  const std::string& getMessage() const { return Message; }

private:
  std::string Message;
};

// Usage
Error doSomething() {
  if (/* error condition */)
    return make_error<CustomError>("Something went wrong");
  return Error::success();
}
```

## Error Handling Patterns

### Basic Error Propagation
```cpp
Error processFile(StringRef Path) {
  if (auto Err = openFile(Path))
    return Err;  // Propagate error

  if (auto Err = validateFormat(Path))
    return Err;

  return Error::success();
}
```

### Expected<T> Usage
```cpp
Expected<Data> loadData(StringRef Path) {
  auto FileOrErr = openFile(Path);
  if (auto Err = FileOrErr.takeError())
    return Err;

  return parseData(*FileOrErr);
}

// Alternative form
Expected<Data> loadData(StringRef Path) {
  if (auto FileOrErr = openFile(Path)) {
    auto& File = *FileOrErr;
    return parseData(File);
  } else {
    return FileOrErr.takeError();
  }
}
```

### Error Consumption
Error values are most commonly passed up the stack (having interrupted whatever
operation raised the error). Eventually errors must be consumed (failure to do
so will trigger an assertion). Errors may be consumed using one of the
following patterns:

```cpp
// 1. Handle specific error types
handleAllErrors(mayFail(),
  [](const CustomError& CE) {
    // Handle CustomError
  },
  [](ErrorInfoBase& EIB) {
    // Handle any other error
  }
);

// 2. Report errors to the Session:
//    This should be done for Errors that cannot be passed further up the stack
//    (e.g. the have reached the root of some thread)
{
  if (auto Err = mayFail())
    S.reportError(std::move(Err));
  // thread ends here.
}

// 3. Convert to string and log:
//    This option may be used in contexts where a reference to the Session is
//    not available.
logError(toString(mayFail()));

// 4  Consume and ignore (explicit)
//    Errors can be explicitly consumed in cases where a failure is known to be
//    benign.
if (auto Err = tryPopulateFromOnDiskCache(...))
  consumeError(std::move(Err)); // Error indicates cache unavailable. Benign.
```

## Exception Interoperability

When `ORC_RT_ENABLE_EXCEPTIONS=On`, ORC-RT provides bidirectional conversion
between errors and exceptions.

> **Important: Exception Usage Policy**
>
> **ORC-RT should not use exceptions internally.** All ORC-RT functions should
> use `Error` and `Expected<T>` return types for error reporting. Exceptions
> should only be used at the boundaries:
>
> 1. **Converting external exceptions to errors** when calling
>      exception-throwing external code
> 2. **Converting errors to exceptions** when returning from ORC-RT to
>      exception-expecting client code
>
> This policy ensures that:
> - ORC-RT works consistently whether exceptions are enabled or disabled
> - Error handling behavior is predictable and doesn't depend on exception
>   propagation
> - The library remains compatible with codebases that disable exceptions
>   (most LLVM projects)

### Core Interop APIs

**`runCapturingExceptions`**: Converts exceptions to errors
```cpp
// Return type depends on callback:
//   void        → Error
//   Error       → Error
//   Expected<T> → Expected<T>
//   T           → Expected<T>

auto Result = runCapturingExceptions([]() {
  return riskyOperation();  // might throw
});
```

**`Error::throwOnFailure`**: Converts errors to exceptions
```cpp
try {
  auto Err = orcOperation();
  Err.throwOnFailure();  // Throws if Err represents failure
} catch (std::unique_ptr<StringError>& E) {
  // Catch specific error types
} catch (std::unique_ptr<ErrorInfoBase>& E) {
  // Catch any ORC error
} catch (...) {
  // Catch other exceptions
}
```

### Exception Boundary Pattern

Use `runCapturingExceptions` to prevent exceptions from unwinding through ORC
runtime:

```cpp
Error safeCallback(std::function<void()> UserCallback) {
  return runCapturingExceptions([&]() {
    UserCallback();  // User code might throw
  });
}
```

### ExceptionError Type

`ExceptionError` preserves C++ exceptions as `Error` values:

```cpp
auto Err = runCapturingExceptions([]() {
  throw std::runtime_error("C++ exception");
});

// Err contains an ExceptionError wrapping the std::runtime_error
assert(Err.isA<ExceptionError>());

// Can be rethrown with original type preserved
Err.throwOnFailure();  // Rethrows std::runtime_error
```

## Best Practices

### 1. Consistent Return Types
```cpp
// Good: Consistent error handling
Expected<Data> loadData(StringRef Path);
Error saveData(const Data& D, StringRef Path);

// Bad: Mixed error handling
Data loadDataOrDie(StringRef Path);  // Inconsistent
bool saveData(const Data& D, StringRef Path, std::string* Error);  // C-style
```

### 2. Meaningful Error Messages
```cpp
// Good: Descriptive, actionable
return make_error<StringError>(
  "Failed to parse config file '" + Path + "': invalid JSON at line " +
  std::to_string(LineNum)
);

// Bad: Vague
return make_error<StringError>("Parse error");
```

### 3. Appropriate Error Granularity
```cpp
// Good: Specific error types enable targeted handling
class FileNotFoundError : public ErrorExtends<FileNotFoundError, ErrorInfoBase> {
  // ... specific to missing files
};

class PermissionError : public ErrorExtends<PermissionError, ErrorInfoBase> {
  // ... specific to permission issues
};

// Usage allows specific handling
handleAllErrors(openFile(Path),
  [](const FileNotFoundError& E) { /* try alternative locations */ },
  [](const PermissionError& E)   { /* request elevated access */ },
  [](ErrorInfoBase& E)           { /* generic fallback */ }
);
```

### 4. Exception Safety in Mixed Environments
```cpp
// Safe pattern: Isolate exception-throwing code
Error integrateWithExceptionThrowingLibrary() {
  return runCapturingExceptions([&]() {
    externalLibrary.riskyOperation();
    return Error::success();
  });
}

// Unsafe: Exceptions can unwind through Error values
Error unsafeIntegration() {
  if (auto Err = orcOperation()) {
    log("Failed");  // might throw!
    return Err;     // ASSERTION FAILURE if log() throws
  }
  return Error::success();
}
```

### 5. Performance Considerations
- `Error::success()` is zero-cost
- Avoid creating error objects in hot paths when possible
- Use early returns to minimize deep nesting

```cpp
// Good: Early return, minimal overhead
Error fastPath(bool condition) {
  if (ORC_RT_LIKELY(condition))
    return Error::success();

  return make_error<StringError>("Rare error case");
}
```

## Configuration Impact

### Exception Disabled (`ORC_RT_ENABLE_EXCEPTIONS=Off`)
- `throwOnFailure()` and `runCapturingExceptions()` are not available
- `ExceptionError` is not available
- All error handling uses `Error`/`Expected<T>` exclusively
- Compatible with LLVM projects that disable exceptions

### Exceptions Enabled (`ORC_RT_ENABLE_EXCEPTIONS=On`)
- Full interoperability between errors and exceptions
- Safe integration with exception-throwing external libraries
- `ExceptionError` preserves exception values across Error boundaries
- Compatible with standard C++ codebases using exceptions

## Summary

ORC-RT's error handling system provides:

- **Type Safety**: Errors have specific types that can be handled appropriately
- **Performance**: Zero-cost success path, efficient error propagation
- **Flexibility**: Works with or without C++ exceptions
- **Interoperability**: Supports integration with exception-throwing code
- **Consistency**: Uniform error handling across the entire codebase

By following these guidelines, ORC-RT maintains robust error handling that works
across diverse integration environments while providing clear, actionable error
information to users and developers.
