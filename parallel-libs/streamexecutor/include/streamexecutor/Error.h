//===-- Error.h - Error handling --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Error types used in the public API and internally in StreamExecutor.
///
/// StreamExecutor's error handling is based on the types streamexecutor::Error
/// and streamexecutor::Expected<T>.
///
///
/// \section error The Error Class
///
/// The Error class either represents success or contains an error message
/// describing the cause of the error. Error instances are created by calling
/// Error::success for successes or make_error for errors.
///
/// \code{.cpp}
/// Error achieveWorldPeace() {
///   if (WorldPeaceAlreadyAchieved) {
///     return Error::success();
///   } else {
///     return make_error("Can't someone else do it?");
///   }
/// }
/// \endcode
///
/// Error instances are implicitly convertible to bool. Error values convert to
/// true and successes convert to false. Error instances must have their boolean
/// values checked or they must be moved before they go out of scope, otherwise
/// their destruction will cause the program to abort with a warning about an
/// unchecked Error.
///
/// If the Error represents success, then checking the boolean value is all that
/// is required, but if the Error represents a real error, the Error value must
/// be consumed. The function consumeAndGetMessage is the way to extract the
/// error message from an Error and consume the Error at the same time, so
/// typical error handling will first check whether there was an error and then
/// extract the error message if so. Here is an example:
///
/// \code{.cpp}
/// if (Error E = achieveWorldPeace()) {
///   printf("An error occurred: %s\n", consumeAndGetMessage(E).c_str());
///   exit(EXIT_FAILURE):
/// }
/// \endcode
///
/// It is also common to simply pass an error along up the call stack if it
/// cannot be handled in the current function.
///
/// \code{.cpp}
/// Error doTask() {
///   if (Error E = achieveWorldPeace()) {
///     return E;
///   }
///   ...
/// }
/// \endcode
///
/// There is also a function consumeError that consumes an error value without
/// fetching the error message. This is useful when we want to ignore an error.
///
/// The dieIfError function is also provided for quick-and-dirty error handling.
///
///
/// \section expected The Expected Class
///
/// The Expected<T> class either represents a value of type T or an Error.
/// Expected<T> has one constructor that takes a T value and another constructor
/// that takes an Error rvalue reference, so Expected instances can be
/// constructed either from values or from errors:
///
/// \code{.cpp}
/// Expected<int> getMyFavoriteInt() {
///   int MyFavorite = 42;
///   if (IsThereAFavorite) {
///     return MyFavorite;
///   } else {
///     return make_error("I don't have a favorite");
///   }
/// }
/// \endcode
///
/// Expected<T> instances are implicitly convertible to bool and are true if
/// they contain a value and false if they contain an error. Note that this is
/// the opposite convention of the Error type conversion to bool, where true
/// meant error and false meant success.
///
/// If the Expected<T> instance is not an error, the stored value can be
/// obtained by using operator*. If access to members of the value are desired
/// instead of the value itself, operator-> can be used as well.
///
/// Expected<T> instances must have their boolean value checked or they must be
/// moved before they go out of scope, otherwise they will cause the program to
/// abort with a warning about an unchecked error. If the Expected<T> instance
/// contains a value, then checking the boolean value is all that is required,
/// but if it contains an Error object, that Error object must be handled by
/// calling Expected<T>::takeError() to get the underlying error.
///
/// Here is an example of the use of an Expected<T> value returned from a
/// function:
///
/// \code{.cpp}
/// Expected<int> ExpectedInt = getMyFavoriteInt();
/// if (ExpectedInt) {
///   printf("My favorite integer is %d\n", *ExpectedInt);
/// } else {
///   printf("An error occurred: %s\n",
///     consumeAndGetMessage(ExpectedInt.takeError()));
///   exit(EXIT_FAILURE);
/// }
/// \endcode
///
/// The following snippet shows some examples of how Errors and Expected values
/// can be passed up the stack if they should not be handled in the current
/// function.
///
/// \code{.cpp}
/// Expected<double> doTask3() {
///   Error WorldPeaceError = achieveWorldPeace();
///   if (!WorldPeaceError) {
///     return WorldPeaceError;
///   }
///
///   Expected<martian> ExpectedMartian = getMyFavoriteMartian();
///   if (!ExpectedMartian) {
///     // Must extract the error because martian cannot be converted to double.
///     return ExpectedMartian.takeError():
///   }
///
///   // It's fine to return Expected<int> for Expected<double> because int can
///   // be converted to double.
///   return getMyFavoriteInt();
/// }
/// \endcode
///
/// The getOrDie function is also available for quick-and-dirty error handling.
///
///
/// \section llvm Relation to llvm::Error and llvm::Expected
///
/// The streamexecutor::Error and streamexecutor::Expected classes are actually
/// just their LLVM counterparts redeclared in the streamexectuor namespace, but
/// they should be treated as separate types, even so.
///
/// StreamExecutor does not support any underlying llvm::ErrorInfo class except
/// the one it defines internally for itself, so a streamexecutor::Error can be
/// thought of as a restricted llvm::Error that is guaranteed to hold a specific
/// error type.
///
/// Although code may compile if llvm functions used to handle these
/// StreamExecutor error types, it is likely that code will lead to runtime
/// errors, so it is strongly recommended that only the functions from the
/// streamexecutor namespace are used on these StreamExecutor error types.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_ERROR_H
#define STREAMEXECUTOR_ERROR_H

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "llvm/Support/Error.h"

namespace streamexecutor {

using llvm::consumeError;
using llvm::Error;
using llvm::Expected;
using llvm::Twine;

/// Makes an Error object from an error message.
Error make_error(Twine Message);

/// Consumes the input error and returns its error message.
///
/// Assumes the input was created by the make_error function above.
std::string consumeAndGetMessage(Error &&E);

/// Extracts the T value from an Expected<T> or prints an error message to
/// stderr and exits the program with code EXIT_FAILURE if the Expected<T> is an
/// error.
///
/// This function and the dieIfError function are provided for applications that
/// are OK with aborting the program if an error occurs, and which don't have
/// any special error logging needs. Applications with different error handling
/// needs will likely want to declare their own functions with similar
/// signatures but which log error messages in a different way or attempt to
/// recover from errors instead of aborting the program.
template <typename T> T getOrDie(Expected<T> &&E) {
  if (!E) {
    std::fprintf(stderr, "Error extracting an expected value: %s.\n",
                 consumeAndGetMessage(E.takeError()).c_str());
    std::exit(EXIT_FAILURE);
  }
  return std::move(*E);
}

/// Prints an error message to stderr and exits the program with code
/// EXIT_FAILURE if the input is an error.
///
/// \sa getOrDie
void dieIfError(Error &&E);

} // namespace streamexecutor

#endif // STREAMEXECUTOR_ERROR_H
