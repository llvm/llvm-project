// test functions for callbacks that are passed onto ompd_initialize as a callback table
// copy into ompdModule.c and insert in ompd_open if necessary
static void testAlloc(int size);

static void testRead(void);

static void testThreadContext(void);

static void testSymAddr(const char* evalSymbol)
{
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        PyObject* pFunc = PyObject_GetAttrString(pModule, "_sym_addr");
        if(pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = PyTuple_New(2);
                PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 1));
                PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", evalSymbol));
                PyObject* returnVal = PyObject_CallObject(pFunc, pArgs);
                PyObject* printFunc = PyObject_GetAttrString(pModule, "_print");
                PyObject* printArgs = PyTuple_New(1);
                PyTuple_SetItem(printArgs, 0, returnVal);
                PyObject_CallObject(printFunc, printArgs);
        }
}

static void testPrint(const char* printString)
{
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        PyObject* pFunc = PyObject_GetAttrString(pModule, "_print");
        if(pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = PyTuple_New(1);
                PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", printString));
                PyObject_CallObject(pFunc, pArgs);
        }
}

static void testThreadObjects(void)
{
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        PyObject* pFunc = PyObject_GetAttrString(pModule, "_test_threads");
        if(pFunc && PyCallable_Check(pFunc)) {
                PyObject_CallObject(pFunc, NULL);
        }
}

static void testAlloc(int size)
{
        int* field ;
        _alloc(size, (void**)&field);
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        PyObject* printFunc = PyObject_GetAttrString(pModule, "_print");
        PyObject* printArgs = PyTuple_New(1);
        PyTuple_SetItem(printArgs, 0, Py_BuildValue("i", field[size-1]));
        PyObject_CallObject(printFunc, printArgs);
}

static void testRead(void)
{
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        // address to read
        char* buffer = malloc(sizeof(char)*40);
        void* buf = (void*) buffer;
        int i = 0;
        for(i = 0; i < 40; i++) {
                buffer[i] = 0;
        }

        ompd_address_t myAddress = { ((ompd_seg_t)0), 0 };
        _sym_addr(NULL, NULL, "ompd_state", &myAddress);
        char tmp[200];
        sprintf(tmp, "Symbol 0x%lx\n", myAddress.address);
         _print(tmp);
        _read(NULL, NULL, myAddress, 8, buf);
        sprintf(tmp, "Content: %lli\n", *(long long int*)buffer);
         _print(tmp);
}

static void testThreadContext(void)
{
        if(pModule == NULL) {
                pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
        }
        int kind = 0;
        long int address = 7l;
        PyObject* threadFunc = PyObject_GetAttrString(pModule, "_thread_context");
        PyObject* threadArgs = PyTuple_New(2);
        PyTuple_SetItem(threadArgs, 0, Py_BuildValue("i", kind));
        PyTuple_SetItem(threadArgs, 1, Py_BuildValue("l", address));
        PyObject_CallObject(threadFunc, threadArgs);
}
