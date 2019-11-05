#include "PythonReadline.h"

#ifdef LLDB_USE_LIBEDIT_READLINE_COMPAT_MODULE

#include <stdio.h>

#include <editline/readline.h>

// Simple implementation of the Python readline module using libedit.
// In the event that libedit is excluded from the build, this turns
// back into a null implementation that blocks the module from pulling
// in the GNU readline shared lib, which causes linkage confusion when
// both readline and libedit's readline compatibility symbols collide.
//
// Currently it only installs a PyOS_ReadlineFunctionPointer, without
// implementing any of the readline module methods. This is meant to
// work around LLVM pr18841 to avoid seg faults in the stock Python
// readline.so linked against GNU readline.
//
// Bug on the cpython side: https://bugs.python.org/issue38634

PyDoc_STRVAR(moduleDocumentation,
             "Simple readline module implementation based on libedit.");

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef readline_module = {
    PyModuleDef_HEAD_INIT, // m_base
    "lldb_editline",       // m_name
    moduleDocumentation,   // m_doc
    -1,                    // m_size
    nullptr,               // m_methods
    nullptr,               // m_reload
    nullptr,               // m_traverse
    nullptr,               // m_clear
    nullptr,               // m_free
};
#else
static struct PyMethodDef moduleMethods[] = {{nullptr, nullptr, 0, nullptr}};
#endif

static char *
#if PY_MAJOR_VERSION >= 3
simple_readline(FILE *stdin, FILE *stdout, const char *prompt)
#else
simple_readline(FILE *stdin, FILE *stdout, char *prompt)
#endif
{
  rl_instream = stdin;
  rl_outstream = stdout;
  char *line = readline(prompt);
  if (!line) {
#if PY_MAJOR_VERSION >= 3
    char *ret = (char *)PyMem_RawMalloc(1);
#else
    char *ret = (char *)PyMem_Malloc(1);
#endif
    if (ret != NULL)
      *ret = '\0';
    return ret;
  }
  if (*line)
    add_history(line);
  int n = strlen(line);
#if PY_MAJOR_VERSION >= 3
  char *ret = (char *)PyMem_RawMalloc(n + 2);
#else
  char *ret = (char *)PyMem_Malloc(n + 2);
#endif
  if (ret) {
    strncpy(ret, line, n);
    free(line);
    ret[n] = '\n';
    ret[n + 1] = '\0';
  }
  return ret;
}

PyMODINIT_FUNC initlldb_readline(void) {
  PyOS_ReadlineFunctionPointer = simple_readline;

#if PY_MAJOR_VERSION >= 3
  return PyModule_Create(&readline_module);
#else
  Py_InitModule4("readline", moduleMethods, moduleDocumentation,
                 static_cast<PyObject *>(NULL), PYTHON_API_VERSION);
#endif
}
#endif
