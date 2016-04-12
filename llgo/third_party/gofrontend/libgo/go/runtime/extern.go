// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package runtime contains operations that interact with Go's runtime system,
such as functions to control goroutines. It also includes the low-level type information
used by the reflect package; see reflect's documentation for the programmable
interface to the run-time type system.

Environment Variables

The following environment variables ($name or %name%, depending on the host
operating system) control the run-time behavior of Go programs. The meanings
and use may change from release to release.

The GOGC variable sets the initial garbage collection target percentage.
A collection is triggered when the ratio of freshly allocated data to live data
remaining after the previous collection reaches this percentage. The default
is GOGC=100. Setting GOGC=off disables the garbage collector entirely.
The runtime/debug package's SetGCPercent function allows changing this
percentage at run time. See https://golang.org/pkg/runtime/debug/#SetGCPercent.

The GODEBUG variable controls debugging variables within the runtime.
It is a comma-separated list of name=val pairs setting these named variables:

	allocfreetrace: setting allocfreetrace=1 causes every allocation to be
	profiled and a stack trace printed on each object's allocation and free.

	efence: setting efence=1 causes the allocator to run in a mode
	where each object is allocated on a unique page and addresses are
	never recycled.

	gccheckmark: setting gccheckmark=1 enables verification of the
	garbage collector's concurrent mark phase by performing a
	second mark pass while the world is stopped.  If the second
	pass finds a reachable object that was not found by concurrent
	mark, the garbage collector will panic.

	gcpacertrace: setting gcpacertrace=1 causes the garbage collector to
	print information about the internal state of the concurrent pacer.

	gcshrinkstackoff: setting gcshrinkstackoff=1 disables moving goroutines
	onto smaller stacks. In this mode, a goroutine's stack can only grow.

	gcstackbarrieroff: setting gcstackbarrieroff=1 disables the use of stack barriers
	that allow the garbage collector to avoid repeating a stack scan during the
	mark termination phase.

	gcstackbarrierall: setting gcstackbarrierall=1 installs stack barriers
	in every stack frame, rather than in exponentially-spaced frames.

	gcstoptheworld: setting gcstoptheworld=1 disables concurrent garbage collection,
	making every garbage collection a stop-the-world event. Setting gcstoptheworld=2
	also disables concurrent sweeping after the garbage collection finishes.

	gctrace: setting gctrace=1 causes the garbage collector to emit a single line to standard
	error at each collection, summarizing the amount of memory collected and the
	length of the pause. Setting gctrace=2 emits the same summary but also
	repeats each collection. The format of this line is subject to change.
	Currently, it is:
		gc # @#s #%: #+...+# ms clock, #+...+# ms cpu, #->#-># MB, # MB goal, # P
	where the fields are as follows:
		gc #        the GC number, incremented at each GC
		@#s         time in seconds since program start
		#%          percentage of time spent in GC since program start
		#+...+#     wall-clock/CPU times for the phases of the GC
		#->#-># MB  heap size at GC start, at GC end, and live heap
		# MB goal   goal heap size
		# P         number of processors used
	The phases are stop-the-world (STW) sweep termination, scan,
	synchronize Ps, mark, and STW mark termination. The CPU times
	for mark are broken down in to assist time (GC performed in
	line with allocation), background GC time, and idle GC time.
	If the line ends with "(forced)", this GC was forced by a
	runtime.GC() call and all phases are STW.

	memprofilerate: setting memprofilerate=X will update the value of runtime.MemProfileRate.
	When set to 0 memory profiling is disabled.  Refer to the description of
	MemProfileRate for the default value.

	memprofilerate:  setting memprofilerate=X changes the setting for
	runtime.MemProfileRate.  Refer to the description of this variable for how
	it is used and its default value.

	sbrk: setting sbrk=1 replaces the memory allocator and garbage collector
	with a trivial allocator that obtains memory from the operating system and
	never reclaims any memory.

	scavenge: scavenge=1 enables debugging mode of heap scavenger.

	scheddetail: setting schedtrace=X and scheddetail=1 causes the scheduler to emit
	detailed multiline info every X milliseconds, describing state of the scheduler,
	processors, threads and goroutines.

	schedtrace: setting schedtrace=X causes the scheduler to emit a single line to standard
	error every X milliseconds, summarizing the scheduler state.

The GOMAXPROCS variable limits the number of operating system threads that
can execute user-level Go code simultaneously. There is no limit to the number of threads
that can be blocked in system calls on behalf of Go code; those do not count against
the GOMAXPROCS limit. This package's GOMAXPROCS function queries and changes
the limit.

The GOTRACEBACK variable controls the amount of output generated when a Go
program fails due to an unrecovered panic or an unexpected runtime condition.
By default, a failure prints a stack trace for every extant goroutine, eliding functions
internal to the run-time system, and then exits with exit code 2.
If GOTRACEBACK=0, the per-goroutine stack traces are omitted entirely.
If GOTRACEBACK=1, the default behavior is used.
If GOTRACEBACK=2, the per-goroutine stack traces include run-time functions.
If GOTRACEBACK=crash, the per-goroutine stack traces include run-time functions,
and if possible the program crashes in an operating-specific manner instead of
exiting. For example, on Unix systems, the program raises SIGABRT to trigger a
core dump.

The GOARCH, GOOS, GOPATH, and GOROOT environment variables complete
the set of Go environment variables. They influence the building of Go programs
(see https://golang.org/cmd/go and https://golang.org/pkg/go/build).
GOARCH, GOOS, and GOROOT are recorded at compile time and made available by
constants or functions in this package, but they do not influence the execution
of the run-time system.
*/
package runtime

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched()

// Goexit terminates the goroutine that calls it.  No other goroutine is affected.
// Goexit runs all deferred calls before terminating the goroutine.
//
// Calling Goexit from the main goroutine terminates that goroutine
// without func main returning. Since func main has not returned,
// the program continues execution of other goroutines.
// If all other goroutines exit, the program crashes.
func Goexit()

// Caller reports file and line number information about function invocations on
// the calling goroutine's stack.  The argument skip is the number of stack frames
// to ascend, with 0 identifying the caller of Caller.  (For historical reasons the
// meaning of skip differs between Caller and Callers.) The return values report the
// program counter, file name, and line number within the file of the corresponding
// call.  The boolean ok is false if it was not possible to recover the information.
func Caller(skip int) (pc uintptr, file string, line int, ok bool)

// Callers fills the slice pc with the program counters of function invocations
// on the calling goroutine's stack.  The argument skip is the number of stack frames
// to skip before recording in pc, with 0 identifying the frame for Callers itself and
// 1 identifying the caller of Callers.
// It returns the number of entries written to pc.
func Callers(skip int, pc []uintptr) int

type Func struct {
	opaque struct{} // unexported field to disallow conversions
}

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
func FuncForPC(pc uintptr) *Func

// Name returns the name of the function.
func (f *Func) Name() string {
	return funcname_go(f)
}

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr {
	return funcentry_go(f)
}

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	return funcline_go(f, pc)
}

// implemented in symtab.c
func funcline_go(*Func, uintptr) (string, int)
func funcname_go(*Func) string
func funcentry_go(*Func) uintptr

// SetFinalizer sets the finalizer associated with x to f.
// When the garbage collector finds an unreachable block
// with an associated finalizer, it clears the association and runs
// f(x) in a separate goroutine.  This makes x reachable again, but
// now without an associated finalizer.  Assuming that SetFinalizer
// is not called again, the next time the garbage collector sees
// that x is unreachable, it will free x.
//
// SetFinalizer(x, nil) clears any finalizer associated with x.
//
// The argument x must be a pointer to an object allocated by
// calling new or by taking the address of a composite literal.
// The argument f must be a function that takes a single argument
// to which x's type can be assigned, and can have arbitrary ignored return
// values. If either of these is not true, SetFinalizer aborts the
// program.
//
// Finalizers are run in dependency order: if A points at B, both have
// finalizers, and they are otherwise unreachable, only the finalizer
// for A runs; once A is freed, the finalizer for B can run.
// If a cyclic structure includes a block with a finalizer, that
// cycle is not guaranteed to be garbage collected and the finalizer
// is not guaranteed to run, because there is no ordering that
// respects the dependencies.
//
// The finalizer for x is scheduled to run at some arbitrary time after
// x becomes unreachable.
// There is no guarantee that finalizers will run before a program exits,
// so typically they are useful only for releasing non-memory resources
// associated with an object during a long-running program.
// For example, an os.File object could use a finalizer to close the
// associated operating system file descriptor when a program discards
// an os.File without calling Close, but it would be a mistake
// to depend on a finalizer to flush an in-memory I/O buffer such as a
// bufio.Writer, because the buffer would not be flushed at program exit.
//
// It is not guaranteed that a finalizer will run if the size of *x is
// zero bytes.
//
// A single goroutine runs all finalizers for a program, sequentially.
// If a finalizer must run for a long time, it should do so by starting
// a new goroutine.
func SetFinalizer(x, f interface{})

func getgoroot() string

// GOROOT returns the root of the Go tree.
// It uses the GOROOT environment variable, if set,
// or else the root used during the Go build.
func GOROOT() string {
	s := getgoroot()
	if s != "" {
		return s
	}
	return defaultGoroot
}

// Version returns the Go tree's version string.
// It is either the commit hash and date at the time of the build or,
// when possible, a release tag like "go1.3".
func Version() string {
	return theVersion
}

// GOOS is the running program's operating system target:
// one of darwin, freebsd, linux, and so on.
const GOOS string = theGoos

// GOARCH is the running program's architecture target:
// 386, amd64, arm, arm64, ppc64, ppc64le.
const GOARCH string = theGoarch

// GCCGOTOOLDIR is the Tool Dir for the gccgo build
const GCCGOTOOLDIR string = theGccgoToolDir
