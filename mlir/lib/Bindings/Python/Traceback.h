/* Copyright 2020 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_TRACEBACK_H_
#define JAXLIB_TRACEBACK_H_

#include <Python.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "nanobind/nanobind.h"

namespace mlir::python {

class Traceback : public nanobind::object {
public:
  NB_OBJECT(Traceback, nanobind::object, "Traceback", Traceback::Check);

  // Returns a traceback if it is enabled, otherwise returns nullopt.
  static std::optional<Traceback> Get();

  // Returns a string representation of the traceback.
  std::string ToString() const;

  // Returns a list of (code, lasti) pairs for each frame in the traceback.
  std::vector<std::pair<PyCodeObject *, int>> RawFrames() const;

  struct Frame {
    nanobind::str file_name;
    nanobind::str function_name;
    int function_start_line;
    int line_num;

    std::string ToString() const;
  };
  // Returns a list of Frames for the traceback.
  std::vector<Frame> Frames() const;

private:
  static bool Check(PyObject *o);
};

void BuildTracebackSubmodule(nanobind::module_ &m);

} // namespace mlir::python

#endif // JAXLIB_TRACEBACK_H_
