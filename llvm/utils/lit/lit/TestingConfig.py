import os

class TestingConfig:
    """"
    TestingConfig - Information on the tests inside a suite.
    """

    @staticmethod
    def frompath(path, parent, litConfig, mustExist, config = None):
        if config is None:
            # Set the environment based on the command line arguments.
            environment = {
                'LIBRARY_PATH' : os.environ.get('LIBRARY_PATH',''),
                'LD_LIBRARY_PATH' : os.environ.get('LD_LIBRARY_PATH',''),
                'PATH' : os.pathsep.join(litConfig.path +
                                         [os.environ.get('PATH','')]),
                'PATHEXT' : os.environ.get('PATHEXT',''),
                'SYSTEMROOT' : os.environ.get('SYSTEMROOT',''),
                'LLVM_DISABLE_CRT_DEBUG' : '1',
                'PYTHONUNBUFFERED' : '1',
                }

            config = TestingConfig(parent,
                                   name = '<unnamed>',
                                   suffixes = set(),
                                   test_format = None,
                                   environment = environment,
                                   substitutions = [],
                                   unsupported = False,
                                   on_clone = None,
                                   test_exec_root = None,
                                   test_source_root = None,
                                   excludes = [],
                                   available_features = [])

        if os.path.exists(path):
            # FIXME: Improve detection and error reporting of errors in the
            # config file.
            f = open(path)
            cfg_globals = dict(globals())
            cfg_globals['config'] = config
            cfg_globals['lit'] = litConfig
            cfg_globals['__file__'] = path
            try:
                exec f in cfg_globals
            except SystemExit,status:
                # We allow normal system exit inside a config file to just
                # return control without error.
                if status.args:
                    raise
            f.close()
        elif mustExist:
            litConfig.fatal('unable to load config from %r ' % path)

        config.finish(litConfig)
        return config

    def __init__(self, parent, name, suffixes, test_format,
                 environment, substitutions, unsupported, on_clone,
                 test_exec_root, test_source_root, excludes,
                 available_features):
        self.parent = parent
        self.name = str(name)
        self.suffixes = set(suffixes)
        self.test_format = test_format
        self.environment = dict(environment)
        self.substitutions = list(substitutions)
        self.unsupported = unsupported
        self.on_clone = on_clone
        self.test_exec_root = test_exec_root
        self.test_source_root = test_source_root
        self.excludes = set(excludes)
        self.available_features = set(available_features)

    def clone(self, path):
        # FIXME: Chain implementations?
        #
        # FIXME: Allow extra parameters?
        cfg = TestingConfig(self, self.name, self.suffixes, self.test_format,
                            self.environment, self.substitutions,
                            self.unsupported, self.on_clone,
                            self.test_exec_root, self.test_source_root,
                            self.excludes, self.available_features)
        if cfg.on_clone:
            cfg.on_clone(self, cfg, path)
        return cfg

    def finish(self, litConfig):
        """finish() - Finish this config object, after loading is complete."""

        self.name = str(self.name)
        self.suffixes = set(self.suffixes)
        self.environment = dict(self.environment)
        self.substitutions = list(self.substitutions)
        if self.test_exec_root is not None:
            # FIXME: This should really only be suite in test suite config
            # files. Should we distinguish them?
            self.test_exec_root = str(self.test_exec_root)
        if self.test_source_root is not None:
            # FIXME: This should really only be suite in test suite config
            # files. Should we distinguish them?
            self.test_source_root = str(self.test_source_root)
        self.excludes = set(self.excludes)
