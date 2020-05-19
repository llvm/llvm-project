
# Table of Contents

- [Introduction](#Introduction)
- [Installation](#Installation-Visual-Studio-Code)
- [Configurations](#configurations)
	- [Launch Configuration Settings](#launch-configuration-settings)
	- [Attach Configuration Settings](#attach-configuration-settings)
	- [Example configurations](#example-configurations)
		- [Launching](#launching)
		- [Attach to process using process ID](#attach-using-pid)
		- [Attach to process by name](#attach-by-name)
		- [Loading a core file](#loading-a-core-file)

# Introduction

The `lldb-vscode` tool creates a command line tool that implements the [Visual
Studio Code Debug API](https://code.visualstudio.com/docs/extensionAPI/api-debugging).
It can be installed as an extension for the Visual Studio Code and Nuclide IDE.
The protocol is easy to run remotely and also can allow other tools and IDEs to
get a full featured debugger with a well defined protocol.

# Installation for Visual Studio Code

Installing the plug-in involves creating a directory in the `~/.vscode/extensions` folder and copying the package.json file that is in the same directory as this
documentation into it, and copying to symlinking a lldb-vscode binary into
the `bin` directory inside the plug-in directory.

If you want to make a stand alone plug-in that you can send to others on unix systems:

```
$ mkdir -p ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ cp package.json ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0
$ cd ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ cp /path/to/a/built/lldb-vscode .
$ cp /path/to/a/built/liblldb.so .
```


If you want to make a stand alone plug-in that you can send to others on macOS systems:

```
$ mkdir -p ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ cp package.json ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0
$ cd ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ cp /path/to/a/built/lldb-vscode .
$ rsync -av /path/to/a/built/LLDB.framework LLDB.framework
```

You might need to create additional directories for the `liblldb.so` or `LLDB.framework` inside or next to the `bin` folder depending on how the [rpath](https://en.wikipedia.org/wiki/Rpath) is set in your `lldb-vscode` binary. By default the `Debug` builds of LLDB usually includes
the current executable directory in the rpath, so these steps should work for most people.

To create a plug-in that symlinks into your `lldb-vscode` in your build directory:

```
$ mkdir -p ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ cp package.json ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0
$ cd ~/.vscode/extensions/llvm-org.lldb-vscode-0.1.0/bin
$ ln -s /path/to/a/built/lldb-vscode
```

This is handy if you want to debug and develope the `lldb-vscode` executable when adding features or fixing bugs.

# Configurations

Launching to attaching require you to create a [launch configuration](https://code.visualstudio.com/Docs/editor/debugging#_launch-configurations). This file
defines arguments that get passed to `lldb-vscode` and the configuration settings
control how the launch or attach happens.

## Launch Configuration Settings

When you launch a program with Visual Studio Code you will need to create a [launch.json](https://code.visualstudio.com/Docs/editor/debugging#_launch-configurations)
file that defines how your program will be run. The JSON configuration file can contain the following `lldb-vscode` specific launch key/value pairs:

|parameter          |type|req |         |
|-------------------|----|:--:|---------|
|**name**           |string|Y| A configuration name that will be displayed in the IDE.
|**type**           |string|Y| Must be "lldb-vscode".
|**request**        |string|Y| Must be "launch".
|**program**        |string|Y| Path to the executable to launch.
|**args**           |[string]|| An array of command line argument strings to be passed to the program being launched.
|**cwd**            |string| | The program working directory.
|**env**            |dictionary| | Environment variables to set when launching the program. The format of each environment variable string is "VAR=VALUE" for environment variables with values or just "VAR" for environment variables with no values.
|**stopOnEntry**    |boolean| | Whether to stop program immediately after launching.
|**initCommands**   |[string]| | LLDB commands executed upon debugger startup prior to creating the LLDB target. Commands and command output will be sent to the debugger console when they are executed.
|**preRunCommands** |[string]| | LLDB commands executed just before launching after the LLDB target has been created. Commands and command output will be sent to the debugger console when they are executed.
|**stopCommands**   |[string]| | LLDB commands executed just after each stop. Commands and command output will be sent to the debugger console when they are executed.
|**exitCommands**   |[string]| | LLDB commands executed when the program exits. Commands and command output will be sent to the debugger console when they are executed.
|**terminateCommands** |[string]| | LLDB commands executed when the debugging session ends. Commands and command output will be sent to the debugger console when they are executed.
|**sourceMap**      |[string[2]]| | Specify an array of path re-mappings. Each element in the array must be a two element array containing a source and destination pathname.
|**debuggerRoot**   | string| |Specify a working directory to use when launching lldb-vscode. If the debug information in your executable contains relative paths, this option can be used so that `lldb-vscode` can find source files and object files that have relative paths.

## Attaching Settings

When attaching to a process using LLDB you can attach in a few ways

1. Attach to an existing process using the process ID
2. Attach to an existing process by name
3. Attach by name by waiting for the next instance of a process to launch

The JSON configuration file can contain the following `lldb-vscode` specific launch key/value pairs:

|parameter          |type    |req |         |
|-------------------|--------|:--:|---------|
|**name**           |string  |Y| A configuration name that will be displayed in the IDE.
|**type**           |string  |Y| Must be "lldb-vscode".
|**request**        |string  |Y| Must be "attach".
|**program**        |string  | | Path to the executable to attach to. This value is optional but can help to resolve breakpoints prior the attaching to the program.
|**pid**            |number  | | The process id of the process you wish to attach to. If **pid** is omitted, the debugger will attempt to attach to the program by finding a process whose file name matches the file name from **porgram**. Setting this value to `${command:pickMyProcess}` will allow interactive process selection in the IDE.
|**stopOnEntry**    |boolean| | Whether to stop program immediately after launching.
|**waitFor**        |boolean | | Wait for the process to launch.
|**initCommands**   |[string]| | LLDB commands executed upon debugger startup prior to creating the LLDB target. Commands and command output will be sent to the debugger console when they are executed.
|**preRunCommands** |[string]| | LLDB commands executed just before launching after the LLDB target has been created. Commands and command output will be sent to the debugger console when they are executed.
|**stopCommands**   |[string]| | LLDB commands executed just after each stop. Commands and command output will be sent to the debugger console when they are executed.
|**exitCommands**   |[string]| | LLDB commands executed when the program exits. Commands and command output will be sent to the debugger console when they are executed.
|**terminateCommands** |[string]| | LLDB commands executed when the debugging session ends. Commands and command output will be sent to the debugger console when they are executed.
|**attachCommands** |[string]| | LLDB commands that will be executed after **preRunCommands** which take place of the code that normally does the attach. The commands can create a new target and attach or launch it however desired. This allows custom launch and attach configurations. Core files can use `target create --core /path/to/core` to attach to core files.


## Example configurations

### Launching

This will launch `/tmp/a.out` with arguments `one`, `two`, and `three` and
adds `FOO=1` and `bar` to the environment:

```javascript
{
  "type": "lldb-vscode",
  "request": "launch",
  "name": "Debug",
  "program": "/tmp/a.out",
  "args": [ "one", "two", "three" ],
  "env": [ "FOO=1", "BAR" ],
}
```

### Attach using PID

This will attach to a process `a.out` whose process ID is 123:

```javascript
{
  "type": "lldb-vscode",
  "request": "attach",
  "name": "Attach to PID",
  "program": "/tmp/a.out",
  "pid": 123
}
```

### Attach by Name

This will attach to an existing process whose base
name matches `a.out`. All we have to do is leave the `pid` value out of the
above configuration:

```javascript
{
  "name": "Attach to Name",
  "type": "lldb-vscode",
  "request": "attach",
  "program": "/tmp/a.out",
}
```

If you want to ignore any existing a.out processes and wait for the next instance
to be launched you can add the "waitFor" key value pair:

```javascript
{
  "name": "Attach to Name (wait)",
  "type": "lldb-vscode",
  "request": "attach",
  "program": "/tmp/a.out",
  "waitFor": true
}
```

This will work as long as the architecture, vendor and OS supports waiting
for processes. Currently MacOS is the only platform that supports this.


### Loading a Core File

This loads the coredump file `/cores/123.core` associated with the program
`/tmp/a.out`:

```javascript
{
  "name": "Load coredump",
  "type": "lldb-vscode",
  "request": "attach",
  "coreFile": "/cores/123.core",
  "program": "/tmp/a.out"
}
```
