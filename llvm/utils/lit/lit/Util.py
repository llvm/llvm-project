import os, sys

def detectCPUs():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(capture(['sysctl', '-n', 'hw.ncpu']))
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1 # Default

def mkdir_p(path):
    """mkdir_p(path) - Make the "path" directory, if it does not exist; this
    will also make directories for any missing parent directories."""
    import errno

    if not path or os.path.exists(path):
        return

    parent = os.path.dirname(path) 
    if parent != path:
        mkdir_p(parent)

    try:
        os.mkdir(path)
    except OSError,e:
        # Ignore EEXIST, which may occur during a race condition.
        if e.errno != errno.EEXIST:
            raise

def capture(args, env=None):
    import subprocess
    """capture(command) - Run the given command (or argv list) in a shell and
    return the standard output."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env)
    out,_ = p.communicate()
    return out

def which(command, paths = None):
    """which(command, [paths]) - Look up the given command in the paths string
    (or the PATH environment variable, if unspecified)."""

    if paths is None:
        paths = os.environ.get('PATH','')

    # Check for absolute match first.
    if os.path.isfile(command):
        return command

    # Would be nice if Python had a lib function for this.
    if not paths:
        paths = os.defpath

    # Get suffixes to search.
    # On Cygwin, 'PATHEXT' may exist but it should not be used.
    if os.pathsep == ';':
        pathext = os.environ.get('PATHEXT', '').split(';')
    else:
        pathext = ['']

    # Search the paths...
    for path in paths.split(os.pathsep):
        for ext in pathext:
            p = os.path.join(path, command + ext)
            if os.path.exists(p):
                return p

    return None

def checkToolsPath(dir, tools):
    for tool in tools:
        if not os.path.exists(os.path.join(dir, tool)):
            return False;
    return True;

def whichTools(tools, paths):
    for path in paths.split(os.pathsep):
        if checkToolsPath(path, tools):
            return path
    return None

def printHistogram(items, title = 'Items'):
    import itertools, math

    items.sort(key = lambda (_,v): v)

    maxValue = max([v for _,v in items])

    # Select first "nice" bar height that produces more than 10 bars.
    power = int(math.ceil(math.log(maxValue, 10)))
    for inc in itertools.cycle((5, 2, 2.5, 1)):
        barH = inc * 10**power
        N = int(math.ceil(maxValue / barH))
        if N > 10:
            break
        elif inc == 1:
            power -= 1

    histo = [set() for i in range(N)]
    for name,v in items:
        bin = min(int(N * v/maxValue), N-1)
        histo[bin].add(name)

    barW = 40
    hr = '-' * (barW + 34)
    print '\nSlowest %s:' % title
    print hr
    for name,value in items[-20:]:
        print '%.2fs: %s' % (value, name)
    print '\n%s Times:' % title
    print hr
    pDigits = int(math.ceil(math.log(maxValue, 10)))
    pfDigits = max(0, 3-pDigits)
    if pfDigits:
        pDigits += pfDigits + 1
    cDigits = int(math.ceil(math.log(len(items), 10)))
    print "[%s] :: [%s] :: [%s]" % ('Range'.center((pDigits+1)*2 + 3),
                                    'Percentage'.center(barW),
                                    'Count'.center(cDigits*2 + 1))
    print hr
    for i,row in enumerate(histo):
        pct = float(len(row)) / len(items)
        w = int(barW * pct)
        print "[%*.*fs,%*.*fs)" % (pDigits, pfDigits, i*barH,
                                   pDigits, pfDigits, (i+1)*barH),
        print ":: [%s%s] :: [%*d/%*d]" % ('*'*w, ' '*(barW-w),
                                          cDigits, len(row),
                                          cDigits, len(items))

