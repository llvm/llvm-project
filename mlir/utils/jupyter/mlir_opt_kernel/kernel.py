# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from subprocess import Popen
import os
import subprocess
import tempfile
import traceback
from ipykernel.kernelbase import Kernel

__version__ = '0.0.1'


def _get_executable():
    """Find the mlir-opt executable."""

    def is_exe(fpath):
        """Returns whether executable file."""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    program = os.environ.get('MLIR_OPT_EXECUTABLE', 'mlir-opt')
    path, name = os.path.split(program)
    # Attempt to get the executable
    if path:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            file = os.path.join(path, name)
            if is_exe(file):
                return file
    raise OSError('mlir-opt not found, please see README')


class MlirOptKernel(Kernel):
    """Kernel using mlir-opt inside jupyter.

    The reproducer syntax (`// configuration:`) is used to run passes. The
    previous result can be referenced to by using `_` (this variable is reset
    upon error). E.g.,

    ```mlir
    // configuration: --pass
    func.func @foo(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> { ... }
    ```

    ```mlir
    // configuration: --next-pass
    _
    ```
    """

    implementation = 'mlir'
    implementation_version = __version__

    language_version = __version__
    language = "mlir"
    language_info = {
        "name": "mlir",
        "codemirror_mode": {
            "name": "mlir"
        },
        "mimetype": "text/x-mlir",
        "file_extension": ".mlir",
        "pygments_lexer": "text"
    }

    @property
    def banner(self):
        """Returns kernel banner."""
        # Just a placeholder.
        return "mlir-opt kernel %s" % __version__

    def __init__(self, **kwargs):
        Kernel.__init__(self, **kwargs)
        self._ = None
        self.executable = None
        self.silent = False

    def get_executable(self):
        """Returns the mlir-opt executable path."""
        if not self.executable:
            self.executable = _get_executable()
        return self.executable

    def process_output(self, output):
        """Reports regular command output."""
        if not self.silent:
            # Send standard output
            stream_content = {'name': 'stdout', 'text': output}
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def process_error(self, output):
        """Reports error response."""
        if not self.silent:
            # Send standard error
            stream_content = {'name': 'stderr', 'text': output}
            self.send_response(self.iopub_socket, 'stream', stream_content)

    def do_execute(self,
                   code,
                   silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):
        """Execute user code using mlir-opt binary."""

        def ok_status():
            """Returns OK status."""
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {}
            }

        def run(code):
            """Run the code by pipeing via filesystem."""
            try:
                inputmlir = tempfile.NamedTemporaryFile(delete=False)
                command = [
                    # Specify input and output file to error out if also
                    # set as arg.
                    self.get_executable(),
                    '--color',
                    inputmlir.name,
                    '-o',
                    '-'
                ]
                if code.startswith('// configuration:'):
                    command.append('--run-reproducer')
                # Simple handling of repeating last line.
                if code.endswith('\n_'):
                    if not self._:
                        raise NameError('No previous result set')
                    code = code[:-1] + self._
                inputmlir.write(code.encode("utf-8"))
                inputmlir.close()
                pipe = Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
                output, errors = pipe.communicate()
                exitcode = pipe.returncode
            finally:
                os.unlink(inputmlir.name)

# Replace temporary filename with placeholder. This takes the very
# remote chance where the full input filename (generated above)
# overlaps with something in the dump unrelated to the file.
            fname = inputmlir.name.encode("utf-8")
            output = output.replace(fname, b"<<input>>")
            errors = errors.replace(fname, b"<<input>>")
            return output, errors, exitcode

        self.silent = silent
        if not code.strip():
            return ok_status()

        try:
            output, errors, exitcode = run(code)

            if exitcode:
                self._ = None
            else:
                self._ = output.decode("utf-8")
        except KeyboardInterrupt:
            return {'status': 'abort', 'execution_count': self.execution_count}
        except Exception as error:
            # Print traceback for local debugging.
            traceback.print_exc()
            self._ = None
            exitcode = 255
            errors = repr(error).encode("utf-8")

        if exitcode:
            content = {'ename': '', 'evalue': str(exitcode), 'traceback': []}

            self.send_response(self.iopub_socket, 'error', content)
            self.process_error(errors.decode("utf-8"))

            content['execution_count'] = self.execution_count
            content['status'] = 'error'
            return content

        if not silent:
            data = {}
            data['text/x-mlir'] = self._
            content = {
                'execution_count': self.execution_count,
                'data': data,
                'metadata': {}
            }
            self.send_response(self.iopub_socket, 'execute_result', content)
            self.process_output(self._)
            self.process_error(errors.decode("utf-8"))
        return ok_status()
